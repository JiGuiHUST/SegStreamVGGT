# -*- coding: utf-8 -*-
"""
SegStreamVGGT: 在 AIC 多个航拍语义分割数据集上训练 query 分割模型

结构模仿 SegVGGT (object queries 读 StreamVGGT 的几何 token -> per-query mask
-> 语义分割), 只用 /data1/luhangze/AIC 下 7 个数据集 + 统一标签闭集 AIC-11。

默认配置 = "冻结 StreamVGGT 主干, 只训练 query 分支 + 语义头" (SegVGGT 风格)。

单卡:
    python train_aic.py --datasets uavid,aeroscapes,dlrsd,potsdam,hdmaps --img-size 518
多卡 (torchrun, 自动 DDP):
    torchrun --nproc_per_node=4 train_aic.py --datasets all --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aic_data.dataset import AICCachedDataset, collate_aic, grid_shape    # noqa: E402
from aic_data.losses import (IoUMeter, RunningLoss, query_aux_loss,        # noqa: E402
                             query_class_assignment, total_query_seg_loss)
from aic_data.registry import DEFAULT_DATASETS, IGNORE_INDEX, AIC_NAME, get_spec  # noqa: E402
from segstreamvggt import SegStreamVGGT                                     # noqa: E402

# ===========================================================================
# 默认超参 (与 DPT_Up / vggt-up 的 train_depth.py 保持一致的风格)
# ===========================================================================
IMG_SIZE = 518
BATCH_SIZE = 6               # 每次前向的样本数 (= crop 数)
EPOCHS = 10
WARMUP_EPOCHS = 1
MAX_TRAIN_SAMPLES = 6000     # 每个 epoch 的样本数 (crops)
MAX_VAL_SAMPLES = 200        # 每个 epoch 验证的样本数
VAL_EVERY = 1
INIT_LR = 3e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 0.05
NUM_WORKERS = 8
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0
NUM_FRAMES = 4               # 送进模型的帧数 S (一次前向 = S 个 crop, StreamVGGT 支持任意帧数)
VAL_NUM_FRAMES = 1           # 验证时每样本帧数 (逐 crop 评测)
FRAMES_MODE = "tile-grid"    # tile-grid | sequence | same-scene | independent
CE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
QUERY_AUX_WEIGHT = 1.0       # 逐 query 辅助监督 (防塔缩): 总权重
QUERY_CLS_WEIGHT = 1.0       #   其中逐 query 分类权重
QUERY_BCE_WEIGHT = 1.0       #   其中逐 query mask BCE 权重
QUERY_DICE_WEIGHT = 1.0      #   其中逐 query mask Dice 权重
LABEL_SMOOTHING = 0.0
STAGE_COVERAGE = "0.85,0.5,0"  # 分阶段课程: 先训覆盖率>=85% 的类, 再到 50%, 最后全部
STAGE_EPOCHS = "auto"          # 各阶段 epoch 分配 (auto = 按本阶段新增类数加权)
NUM_QUERIES = 100
QUERY_DIM = 256
OUT_RATIO = 4                # 语义图分辨率 = H/4
LOG_EVERY = 20
AMP_DTYPE = torch.bfloat16
SEED = 1234
LOG_DIR = str(Path(__file__).resolve().parent / "logs_aic")
CACHE_ROOT = str(Path(__file__).resolve().parent / "cache")
BACKBONE_CKPT = "/data/luhangze/aic/StreamVGGT/ckpt/checkpoints.pth"
PRETRAINED_CKPT = None       # 可选的 "已训练好的 SegStreamVGGT" 权重 (含 query 分支)


# ===========================================================================
# 工具
# ===========================================================================
def is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(*a, **kw):
    if is_main():
        print(*a, **kw, flush=True)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world)
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


class DistributedWeightedSampler(DistributedSampler):
    """按样本权重做有放回采样, 并切分到各 rank (每个 epoch 重新采样)。"""

    def __init__(self, weights: np.ndarray, num_samples: int, num_replicas: int, rank: int, seed: int = 0):
        super().__init__(range(len(weights)), num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed)
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples_total = int(num_samples)
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        total = self.num_samples_total * self.num_replicas
        idx = torch.multinomial(self.weights, total, replacement=True, generator=g)
        idx = idx[self.rank * self.num_samples_total: (self.rank + 1) * self.num_samples_total]
        return iter(idx.tolist())

    def __len__(self):
        return self.num_samples_total


# ===========================================================================
# 模型 / 权重
# ===========================================================================
def build_model(args) -> SegStreamVGGT:
    model = SegStreamVGGT(
        img_size=getattr(args, "model_img_size", 518),
        patch_size=14,
        embed_dim=1024,
        num_classes=args.num_classes,
        enable_camera=False,
        enable_point=args.with_geometry,
        enable_depth=args.with_geometry,
        enable_track=False,
        query_num=args.num_queries,
        query_dim=args.query_dim,
        cross_block_layers=args.cross_block_layers,
        cross_block_params={"embed_dim": args.cross_dim, "num_heads": args.cross_heads, "mlp_ratio": 2.0},
        query_self_attention=args.query_self_attention,
        supervise_attend_frame=args.supervise_attend_frame,
        out_ratio=args.out_ratio,
        query_cls_hidden_dim=args.query_cls_hidden_dim,
    )
    return model


def load_checkpoint_into(model: SegStreamVGGT, ckpt_path: str, tag: str, allow_missing_query: bool = True):
    """把 (StreamVGGT 主干 / 之前训练的 SegStreamVGGT) 权重加载进模型。"""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model_sd = model.state_dict()
    matched, shape_mismatch = {}, []
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            matched[k] = v
        elif k in model_sd:
            shape_mismatch.append(k)
    missing = [k for k in model_sd if k not in matched]
    unexpected = [k for k in sd if k not in matched]
    model.load_state_dict(matched, strict=False)

    missing_query = [k for k in missing if ("semantic_head" in k or "query" in k)]
    missing_other = [k for k in missing if k not in missing_query]
    # patch_embed.pos_embed 是按 img_size 插值的 buffer, 训练分辨率不同时形状必然不符,
    # patch_embed 内部 (interpolate_pos_encoding) 会自动双线性重采样, 属正常现象。
    interpolatable = [k for k in missing_other if k.endswith("patch_embed.pos_embed")]
    missing_other = [k for k in missing_other if k not in interpolatable]
    log(f"[CKPT] {tag}: 加载 {len(matched)} 个张量 | missing {len(missing)} "
        f"(其中 query/语义头 {len(missing_query)}), unexpected {len(unexpected)}, 形状不符 {len(shape_mismatch)}")
    if interpolatable:
        log(f"[CKPT]   patch_embed.pos_embed 形状不符 (img_size 不同), 由插值自动处理: {interpolatable}")
    if missing_other:
        log(f"[CKPT]   !! 主干缺失 {len(missing_other)} 个: {missing_other[:6]}")
    if unexpected:
        log(f"[CKPT]   忽略 extra 键 {len(unexpected)}: {unexpected[:4]}")
    if missing_query and not allow_missing_query:
        raise RuntimeError("query 分支权重缺失, 与 --resume/--pretrained-ckpt 不匹配")
    return len(matched), len(missing_other)


def freeze_backbone(model: SegStreamVGGT, freeze: bool = True, scope: str = "seg"):
    """设定可训练范围。

    scope='seg'          : 只训分割部分 (语义头 + object query 分支), **几何重建部分全部冻结**
    scope='seg+geometry' : 分割 + depth/point head (几何头) 一起训
    scope='all'          : 全量微调
    freeze=False         : 不做任何冻结 (调试用)
    """
    if freeze:
        model.set_train_scope(scope)
        model.eval()
    else:
        model.set_train_scope("all")
    report = model.parameter_groups_report()
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[MODEL] train_scope={scope} | 总参数 {n_all/1e6:.1f}M | 可训练 {n_train/1e6:.2f}M "
        f"({100*n_train/n_all:.2f}%) | 冻结 {100*(1-n_train/n_all):.2f}%")
    for label, (tot, tr) in report.items():
        tag = "✅ 可训练" if tr > 0 else "❄️ 冻结"
        log(f"[MODEL]   {label:24s} {tot/1e6:8.2f}M  可训练 {tr/1e6:8.2f}M  {tag}")
    geom_frozen = sum(t for k, (t, tr) in report.items() if "几何" in k and tr == 0)
    if geom_frozen:
        log(f"[MODEL] 几何重建部分已冻结: {geom_frozen/1e6:.2f}M 参数 (depth/point head)")
    return report


def verify_frozen_grads(model, optimizer) -> int:
    """断言: 优化器里不能出现冻结参数的张量, 且冻结参数不应有梯度。返回检查的冻结张量数。"""
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    bad = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) not in trainable_ids:
                bad.append("optimizer 中混入了冻结参数")
    n_frozen = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            continue
        n_frozen += 1
        if p.grad is not None and torch.count_nonzero(p.grad) > 0:
            bad.append(n)
    if bad:
        log(f"[FREEZE] !! 异常: {len(bad)} 处冻结参数出现梯度/被优化器持有: {bad[:5]}")
    return n_frozen


def make_optimizer(model: SegStreamVGGT, args):
    seg_names = model.seg_branch_named_parameters()
    seg_params, bb_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (seg_params if n in seg_names else bb_params).append(p)
    groups = [{"params": seg_params, "lr": args.lr, "name": "query_branch"}]
    if bb_params:
        groups.append({"params": bb_params, "lr": args.lr * args.backbone_lr_mult, "name": "backbone"})
    opt = AdamW(groups, lr=args.lr, betas=BETAS, weight_decay=args.weight_decay)
    log(f"[OPT] 分割分支 lr={args.lr:.2e} ({sum(p.numel() for p in seg_params)/1e6:.2f}M)"
        + (f" | 其它(主干/几何头) lr={args.lr*args.backbone_lr_mult:.2e} "
           f"({sum(p.numel() for p in bb_params)/1e6:.2f}M, x{args.backbone_lr_mult:g})"
           if bb_params else " | 主干与几何头全部冻结 (不参与优化)"))
    return opt


def make_scheduler(optimizer, args, total_steps: int):
    warmup = max(1, int(total_steps * args.warmup_epochs / max(args.epochs, 1)))
    min_ratio = args.min_lr / args.lr

    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return max(min_ratio, 0.5 * (1 + math.cos(math.pi * p)))
    return LambdaLR(optimizer, fn)


# ===========================================================================
# 分阶段课程 (curriculum): 先训"几乎每个数据集都有"的常见类, 再逐步加入稀有的类
# ===========================================================================
def class_coverage(dataset_names, num_classes: int) -> dict:
    """每个类出现在多少个数据集里 (用于分阶段)。"""
    return {c: sum(1 for d in dataset_names if c in get_spec(d).classes)
            for c in range(1, num_classes)}


def build_curriculum(args, dataset_names):
    """返回 [(epochs, [class ids], label), ...]，阶段间的类集合单调递增。

    默认按 `--stage-coverage` 的覆盖率阈值分层 (7 个数据集的实测覆盖率):
        ≥0.85 (6/7): {building, road, vehicle}
        ≥0.50 (4/7): + {tree, low_vegetation}
        =0    (全部): + {sidewalk, parking, terrain, water, person, sky}
    """
    n_ds = max(1, len(dataset_names))
    cov = class_coverage(dataset_names, args.num_classes)
    all_classes = list(range(1, args.num_classes))

    if args.stage_classes:
        stage_sets = []
        for g in str(args.stage_classes).split(";"):
            g = g.strip()
            if not g:
                continue
            stage_sets.append(sorted({int(x) for x in g.split(",") if x.strip()})
                              if g.lower() != "all" else list(all_classes))
    else:
        stage_sets = []
        for t in [float(x) for x in str(args.stage_coverage).split(",")]:
            cls = sorted(c for c in all_classes if cov.get(c, 0) / n_ds >= t - 1e-9)
            if cls:
                stage_sets.append(cls)

    # 去重 + 保证单调递增, 且最后一段覆盖全部类
    merged, prev = [], set()
    for g in stage_sets:
        cur = sorted(set(g) | prev)
        if not cur or (merged and cur == merged[-1]):
            continue
        merged.append(cur)
        prev = set(cur)
    if not merged:
        merged = [all_classes]
    if set(merged[-1]) != set(all_classes):
        merged.append(all_classes)
    merged = [sorted(set(g)) for g in merged]

    # 每个阶段的 epoch 分配: 默认按"本阶段新增的类数"加权
    n_stage = len(merged)
    if str(args.stage_epochs).lower() == "auto":
        weights, prev = [], set()
        for g in merged:
            weights.append(max(1, len(set(g) - prev)))
            prev = set(g)
        tot = sum(weights)
        ep = [max(1, int(round(args.epochs * w / tot))) for w in weights]
        while sum(ep) > args.epochs:            # 超出: 从后往前削到 >=1
            for i in range(len(ep) - 1, -1, -1):
                if sum(ep) <= args.epochs:
                    break
                if ep[i] > 1:
                    ep[i] -= 1
        while sum(ep) < args.epochs:            # 不足: 补到最后一段
            ep[-1] += 1
    else:
        ep = [int(x) for x in str(args.stage_epochs).split(",")]
        ep = (ep + [ep[-1]] * n_stage)[:n_stage] if ep else [args.epochs] * n_stage

    labels, prev = [], set()
    for g in merged:
        added = sorted(set(g) - prev)
        min_cov = min(cov.get(c, 0) for c in g)
        labels.append(f"最低覆盖 {min_cov}/{n_ds} 数据集, 新增 {added}")
        prev = set(g)
    return [(ep[i], merged[i], labels[i]) for i in range(n_stage)]


def set_stage(args, allowed_classes):
    """切换阶段: 更新 loss 的类掩码 + query 的类别分配。"""
    import torch

    allowed = sorted({int(c) for c in allowed_classes if 1 <= int(c) < args.num_classes})
    if not allowed:
        allowed = list(range(1, args.num_classes))
    mask = torch.zeros(args.num_classes, dtype=torch.bool)
    mask[IGNORE_INDEX] = True
    lut = torch.zeros(args.num_classes, dtype=torch.long)      # 非本阶段类 -> 0 (ignore)
    for c in allowed:
        mask[c] = True
        lut[c] = c
    args.stage_class_mask = mask
    args.stage_lut = lut
    args.stage_classes_now = allowed
    args.assigned_cls = query_class_assignment(args.num_queries, args.num_classes, allowed)
    return allowed


# ===========================================================================
# 数据
# ===========================================================================
def stratified_val_indices(val_ds, n_total: int, seed: int = 1234) -> list:
    """按数据集分层抽取验证子集, 保证每个数据集都被评到 (数据量不平衡)。

    先按数据量比例给每个数据集分配额度 (每个数据集至少 1 个), 再轮转交错,
    这样即使 n_total 很小也不会把某个数据集整个丢掉。
    """
    import random

    by_ds = {}
    for i, s in enumerate(val_ds.samples):
        by_ds.setdefault(s.dataset, []).append(i)
    rng = random.Random(seed)
    total = sum(len(v) for v in by_ds.values())
    per_ds = {}
    for name, idxs in by_ds.items():
        k = max(1, int(round(n_total * len(idxs) / max(total, 1))))
        per_ds[name] = rng.sample(idxs, min(k, len(idxs)))

    # 轮转交错: 保证 n_total >= 数据集数时每个数据集至少被抽到一次
    picked, names = [], list(per_ds.keys())
    while len(picked) < n_total and any(per_ds.values()):
        for n in names:
            if per_ds[n]:
                picked.append(per_ds[n].pop(0))
                if len(picked) >= n_total:
                    break
    rng.shuffle(picked)      # 打散顺序, 让 DDP 各 rank 分到的数据集混合
    return picked


def build_loaders(args, rank: int, world: int):
    dataset_names = list(DEFAULT_DATASETS) if args.datasets == "all" else \
        [d.strip() for d in args.datasets.split(",") if d.strip()]

    train_ds = AICCachedDataset(
        cache_root=args.cache_root, datasets=dataset_names, split="train",
        num_frames=args.num_frames, img_size=args.img_size, patch_size=14, train=True,
        frames_mode=args.frames_mode, scale_range=(args.scale_min, args.scale_max),
        hflip_p=args.hflip, vflip_p=args.vflip, rot90_p=args.rot90,
        num_classes=args.num_classes, seed=args.seed, weight_mode=args.dataset_weight,
        drop_empty=not args.keep_empty_labels,
        photometric=args.photometric, rare_boost=args.rare_boost,
    )
    val_ds = AICCachedDataset(
        cache_root=args.cache_root, datasets=dataset_names, split="val",
        num_frames=args.val_num_frames, img_size=args.img_size, patch_size=14, train=False,
        frames_mode=args.frames_mode, val_crops=args.val_crops,
        num_classes=args.num_classes, seed=args.seed,
        drop_empty=not args.keep_empty_labels,
        photometric=False, rare_boost=0.0,
    )

    # --max-samples 是「每 epoch 的 crop 预算」: 一次前向有 num_frames 个 crop,
    # 所以样本数 (sampler 抽的次数) = 预算 / num_frames。0 = 默认全库图片数 × num_frames。
    S_train = max(1, args.num_frames)
    n_crops = args.max_samples if args.max_samples else len(train_ds) * S_train
    n_samples = max(1, int(round(n_crops / S_train)))
    sampler = DistributedWeightedSampler(
        train_ds.sample_weights, num_samples=max(1, n_samples // world), num_replicas=world, rank=rank, seed=args.seed
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
        collate_fn=collate_aic, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    n_val = min(args.max_val_samples, len(val_ds)) if args.max_val_samples else len(val_ds)
    val_idx = stratified_val_indices(val_ds, n_val, seed=args.seed)
    if world > 1:
        val_idx = val_idx[rank::world]
    val_loader = DataLoader(
        torch.utils.data.Subset(val_ds, val_idx), batch_size=args.batch_size, shuffle=False,
        num_workers=max(2, args.num_workers // 2), collate_fn=collate_aic, pin_memory=True,
    )
    log(f"[DATA] 训练池 {len(train_ds)} 张图 | 每 epoch {n_crops} 个 crop = {n_samples} 个样本 "
        f"(每个样本 {S_train} 帧, mode={args.frames_mode}, 每卡 {len(sampler)} 步) | "
        f"验证 {len(val_ds)} 样本 (每卡 {len(val_idx)}, {args.val_num_frames} 帧/样本)")
    log(f"[DATA] 说明: 1 个 crop = 一张图内一块 {train_ds.ch}x{train_ds.cw} 区域"
        + (f"; {args.frames_mode}=同一张图上连续铺开的 {grid_shape(S_train)[0]}x{grid_shape(S_train)[1]} 网格"
           if args.frames_mode == "tile-grid" else "; 帧间关系见 --frames-mode")
        + "; 每 epoch 重新随机抽取 (有放回)")
    for i, name in enumerate(train_ds.dataset_names):
        n_items = len(train_ds.indices_by_dataset[i])
        per_ep = train_ds.dataset_prob[i] * n_crops
        log(f"[DATA]   {name:11s} 图片 {n_items:5d} 张 | 采样占比 {train_ds.dataset_prob[i]*100:5.1f}% "
            f"| 每 epoch ≈ {per_ep:6.0f} crop = {per_ep/max(n_items, 1):5.1f} crop/图 | 类 {get_spec(name).classes}")
    log(f"[DATA] v4: weight_mode={args.dataset_weight} | 含稀有类(sidewalk/parking/water/person)图占比 "
        f"{train_ds.rare_fraction*100:.1f}% | rare_boost={args.rare_boost} | photometric={args.photometric}")
    return train_ds, train_loader, val_ds, val_loader, dataset_names


# ===========================================================================
# 训练 / 验证
# ===========================================================================
def compute_class_weights(train_ds, cap: float = 10.0):
    """median-frequency 类权重 (只用训练池; 0=ignore 权重置 1)。"""
    totals = np.zeros(train_ds.num_classes, dtype=np.float64)
    for d in train_ds.dataset_names:
        for it in train_ds._manifests[d]:
            cp = it.get("class_px")
            if cp:
                totals += np.asarray(cp, dtype=np.float64)
    if totals.sum() <= 0:
        return None
    freq = totals / totals.sum()
    valid = freq > 0
    valid[0] = False
    if not valid.any():
        return None
    med = float(np.median(freq[valid]))
    w = np.ones(train_ds.num_classes, dtype=np.float64)
    w[valid] = med / np.maximum(freq[valid], 1e-9)
    w = np.clip(w, 0.25, float(cap))
    w[0] = 1.0
    return torch.tensor(w, dtype=torch.float32)


def forward_loss(model, batch, args, class_weights=None):
    images = batch["images"].cuda(non_blocking=True)          # B,S,3,H,W
    labels = batch["labels"].cuda(non_blocking=True)          # B,S,H,W
    valid_class = batch["valid_class"].cuda(non_blocking=True)

    # ---- 课程阶段: 只监督本阶段开启的类 ----
    stage_mask = getattr(args, "stage_class_mask", None)
    if stage_mask is not None:
        stage_mask = stage_mask.to(valid_class.device)
        valid_class = valid_class & stage_mask.unsqueeze(0)     # (B,C) 与 C_d 取交集
        lut = args.stage_lut.to(labels.device)
        labels = lut[labels.clamp(0, args.num_classes - 1)]     # 非本阶段类 -> ignore(0)

    # 几何头只在"需要它们参与训练"时才跑: 冻结的 depth/point head 不参与 loss,
    # 因此训练时直接跳过 (省算力); 它们的权重依然在模型里 (可推理时用)。
    need_geometry = bool(args.with_geometry) and args.train_scope == "seg+geometry"
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
        preds = model(images, with_geometry=need_geometry)
    seg_prob = preds["seg_prob"].float()                      # B,C,S,h,w

    if args.loss_res == "feat":
        hw = seg_prob.shape[-2:]
        # labels: (B, S, H, W) -> (B*S, 1, H, W) -> nearest 到特征图分辨率 -> (B, S, h, w)
        target = F.interpolate(
            labels.float().reshape(-1, 1, *labels.shape[-2:]), size=hw, mode="nearest"
        ).reshape(labels.shape[0], labels.shape[1], *hw).long()
        target_q = target
    else:
        # 分割 loss 在原始 518² 上算 (thin class 不被 nearest 抹掉);
        # query 辅助损失仍在特征图分辨率上算 (它与 query_masks 同分辨率)。
        feat_hw = seg_prob.shape[-2:]
        target = labels
        target_q = F.interpolate(
            labels.float().reshape(-1, 1, *labels.shape[-2:]), size=feat_hw, mode="nearest"
        ).reshape(labels.shape[0], labels.shape[1], *feat_hw).long()
        seg_prob = F.interpolate(
            seg_prob.flatten(0, 1), size=labels.shape[-2:], mode="bilinear", align_corners=False
        ).view(labels.shape[0], -1, labels.shape[1], *labels.shape[-2:])

    loss, ce, dice = total_query_seg_loss(
        seg_prob, target, valid_class, ce_weight=args.ce_weight, dice_weight=args.dice_weight,
        ignore_index=IGNORE_INDEX, class_weights=class_weights, label_smoothing=args.label_smoothing,
    )
    parts = {"ce": float(ce), "dice": float(dice)}

    # ---- 逐 query 辅助监督 (防止 query 塔缩, 见 losses.query_aux_loss) ----
    if args.query_aux_weight > 0:
        aux, aux_parts = query_aux_loss(
            preds["query_logits"], preds["query_masks"], target_q, valid_class, args.assigned_cls,
            ignore_index=IGNORE_INDEX, cls_weight=args.query_cls_weight,
            bce_weight=args.query_bce_weight, dice_weight=args.query_dice_weight,
        )
        loss = loss + args.query_aux_weight * aux
        parts.update(aux_parts)
    return loss, parts, preds


def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch, args, class_weights=None):
    model.train()
    if args.freeze_backbone:
        # 主干 (aggregator + 几何头) 冻结: 放 eval 不影响结果 (无 dropout),
        # 但保持显式一致, 可训练分支仍处于 train 模式
        backbone = unwrap(model).aggregator
        backbone.eval()
        for m in (unwrap(model).camera_head, unwrap(model).point_head, unwrap(model).depth_head):
            if m is not None:
                m.eval()
    rl = RunningLoss()
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        loss, parts, preds = forward_loss(model, batch, args, class_weights)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        rl.update(loss.item(), **parts)

        if step == 0:
            # 冻结完整性校验: 冻结参数不应有梯度, 优化器也不应持有它们
            n_frozen = verify_frozen_grads(unwrap(model), optimizer)
            log(f"[FREEZE] 校验通过: {n_frozen} 个冻结张量均无梯度, 优化器仅含可训练参数")
            if args.train_scope == "seg":
                log("[FREEZE] 几何重建部分 (depth/point head + 聚合器主干) 已冻结, 只有"
                    "语义头 + object query 分支在更新")

        if (step + 1) % LOG_EVERY == 0 and is_main():
            lr = scheduler.get_last_lr()[0]
            extra = ""
            for k in ("query_cls", "query_bce", "query_dice"):
                if k in rl.parts:
                    extra += f" {k} {rl.parts[k]/rl.count:.3f}"
            # 塔缩诊断: query 预测的类别数 / mask 平均面积 (当前 batch)
            qcls = preds["query_logits"].float().argmax(dim=-1)
            n_uniq = float(np.mean([len(torch.unique(qcls[b])) for b in range(qcls.shape[0])]))
            m_area = float(torch.sigmoid(preds["query_masks"].float()).mean())
            log(f"  epoch {epoch} step {step+1}/{len(loader)} loss {rl.mean:.4f} "
                f"(ce {rl.parts.get('ce',0)/rl.count:.4f} dice {rl.parts.get('dice',0)/rl.count:.4f}{extra}) "
                f"lr {lr:.2e} [q: {n_uniq:.0f}cls area{m_area:.2f}] {time.time()-t0:.0f}s")
    return rl


@torch.no_grad()
def evaluate(model, loader, args, dataset_names, class_weights=None) -> dict:
    model.eval()
    meter = IoUMeter(args.num_classes, dataset_names, ignore_index=IGNORE_INDEX)
    rl = RunningLoss()
    diag = {"n_unique_cls": [], "mask_mean": []}
    for batch in loader:
        loss, parts, preds = forward_loss(model, batch, args, class_weights)
        rl.update(loss.item(), **parts)
        seg_prob = preds["seg_prob"].float()                  # B,C,S,h,w
        B, C, S, h, w = seg_prob.shape
        labels = batch["labels"]
        valid_class = batch["valid_class"].to(seg_prob.device)
        # query 塔缩诊断
        qcls = preds["query_logits"].float().argmax(dim=-1)                     # B,Q
        diag["n_unique_cls"].append(float(np.mean([len(torch.unique(qcls[b])) for b in range(qcls.shape[0])])))
        diag["mask_mean"].append(float(torch.sigmoid(preds["query_masks"].float()).mean()))
        for s in range(S):
            prob = seg_prob[:, :, s]                          # B,C,h,w
            up = F.interpolate(prob, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            up = up * valid_class[:, :, None, None].to(up.dtype)   # 屏蔽该数据集未标注类
            if C > 1:
                up[:, 0] = -1.0                                   # class 0 = ignore/void, 不作为预测
            pred = up.argmax(dim=1)
            meter.update(pred, labels[:, s], batch["dataset_id"], valid_class)
    # 多卡汇总
    if dist.is_initialized():
        for t in (meter.tp, meter.fp, meter.fn):
            t_cuda = t.cuda()
            dist.all_reduce(t_cuda, op=dist.ReduceOp.SUM)
            t.copy_(t_cuda.cpu())
    valid_classes = {name: list(get_spec(name).classes) for name in dataset_names}
    summary = meter.summary(valid_classes)
    summary["loss"] = rl.mean
    summary["parts"] = rl.mean_parts()
    if diag["n_unique_cls"]:
        summary["query_diag"] = {
            "unique_classes_per_sample": float(np.mean(diag["n_unique_cls"])),
            "mask_area_mean": float(np.mean(diag["mask_mean"])),
        }
    return summary


# ===========================================================================
# main
# ===========================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="SegStreamVGGT AIC 训练")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=INIT_LR)
    ap.add_argument("--min-lr", type=float, default=MIN_LR)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--img-size", default=IMG_SIZE,
                    help="输入尺寸: 518 (正方形) 或 518x728 / 728x728 (HxW, 都需是 14 的倍数)")
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.2)
    ap.add_argument("--pretrained-ckpt", default=PRETRAINED_CKPT,
                    help="已训练的 SegStreamVGGT 权重 (含 query 分支); 否则从 StreamVGGT 主干初始化")
    ap.add_argument("--backbone-ckpt", default=BACKBONE_CKPT)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-samples", type=int, default=MAX_TRAIN_SAMPLES)
    ap.add_argument("--max-val-samples", type=int, default=MAX_VAL_SAMPLES)
    ap.add_argument("--val-crops", type=int, default=4, help="每张验证图最多取几个 crop")
    ap.add_argument("--val-every", type=int, default=VAL_EVERY)
    ap.add_argument("--freeze-backbone", type=int, default=1)
    ap.add_argument("--train-scope", default="seg", choices=["seg", "seg+geometry", "all"],
                    help="可训练范围: seg=只训分割(几何重建部分冻结, 默认) / seg+geometry=连 depth,point head 一起训 / all=全量")
    ap.add_argument("--aggregator-eval", type=int, default=1,
                    help="冻结主干时把 aggregator 设为 eval (无 dropout, 不影响结果)")
    ap.add_argument("--with-geometry", type=int, default=0, help="同时跑 depth/point head (更慢)")
    ap.add_argument("--num-frames", type=int, default=NUM_FRAMES,
                    help="一次前向的帧数 S (一次样本 = S 个 crop, 如 4 = 同图 2x2 网格)")
    ap.add_argument("--val-num-frames", type=int, default=VAL_NUM_FRAMES, help="验证时每样本帧数")
    ap.add_argument("--frames-mode", default=FRAMES_MODE,
                    choices=["tile-grid", "sequence", "same-scene", "independent"])
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--dataset-weight", default="sqrt", choices=["equal", "sqrt", "natural", "valid"],
                    help="数据集间采样权重: equal / sqrt(sqrt(N_d)) / natural(N_d) / valid(sqrt(有效标注像素))")
    ap.add_argument("--photometric", type=int, default=1,
                    help="=1 开启光度增强 (亮度/对比度/gamma/饱和度/噪声/模糊), v4 默认开")
    ap.add_argument("--rare-boost", type=float, default=1.5,
                    help="类感知采样: 每出现一个稀有类 (sidewalk/parking/water/person), 该图权重 +rare_boost")
    ap.add_argument("--class-weights", type=int, default=0,
                    help="=1 用 median-frequency 类权重 (cap 见 --class-weight-cap)")
    ap.add_argument("--class-weight-cap", type=float, default=10.0)
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    ap.add_argument("--num-classes", type=int, default=12)
    ap.add_argument("--num-queries", type=int, default=NUM_QUERIES)
    ap.add_argument("--query-dim", type=int, default=QUERY_DIM)
    ap.add_argument("--cross-dim", type=int, default=256)
    ap.add_argument("--cross-heads", type=int, default=8)
    ap.add_argument("--cross-block-layers", default=None, help="如 '12,15,18,21,23'; 默认全部 global block")
    ap.add_argument("--query-self-attention", type=int, default=0)
    ap.add_argument("--supervise-attend-frame", type=int, default=0)
    ap.add_argument("--query-cls-hidden-dim", type=int, default=64)
    ap.add_argument("--out-ratio", type=int, default=OUT_RATIO)
    ap.add_argument("--ce-weight", type=float, default=CE_WEIGHT)
    ap.add_argument("--dice-weight", type=float, default=DICE_WEIGHT)
    ap.add_argument("--stage-coverage", default=STAGE_COVERAGE,
                    help="分阶段课程: 覆盖率阈值列表 (逗号分隔), 如 '0.85,0.5,0' = 先训几乎所有数据集都有的类")
    ap.add_argument("--stage-epochs", default=STAGE_EPOCHS,
                    help="各阶段 epoch 分配: 'auto'(按新增类数加权) 或 '3,2,7'")
    ap.add_argument("--stage-classes", default=None,
                    help="直接指定各阶段的类 (分号分隔, 如 '1,2,9;1,2,5,6,9;all'), 优先于 --stage-coverage")
    ap.add_argument("--no-curriculum", type=int, default=0, help="=1 时不分阶段 (一次性监督全部类)")
    ap.add_argument("--query-aux-weight", type=float, default=QUERY_AUX_WEIGHT,
                    help="逐 query 辅助监督总权重 (0 = 关闭; 防止 query 塔缩)")
    ap.add_argument("--query-cls-weight", type=float, default=QUERY_CLS_WEIGHT)
    ap.add_argument("--query-bce-weight", type=float, default=QUERY_BCE_WEIGHT)
    ap.add_argument("--query-dice-weight", type=float, default=QUERY_DICE_WEIGHT)
    ap.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    ap.add_argument("--loss-res", default="feat", choices=["feat", "full"])
    ap.add_argument("--keep-empty-labels", type=int, default=0,
                    help="=1 时不跳过『无有效标签像素』的图 (默认跳过, 如 AeroScapes 只有 thing 类的帧)")
    ap.add_argument("--scale-min", type=float, default=0.5)
    ap.add_argument("--scale-max", type=float, default=2.0)
    ap.add_argument("--hflip", type=float, default=0.5)
    ap.add_argument("--vflip", type=float, default=0.2)
    ap.add_argument("--rot90", type=float, default=0.75)
    ap.add_argument("--save-trainable-only", type=int, default=0)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--log-dir", default=LOG_DIR)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.cross_block_layers:
        args.cross_block_layers = [int(x) for x in str(args.cross_block_layers).split(",")]
    for k in ("freeze_backbone", "with_geometry", "query_self_attention", "supervise_attend_frame",
              "amp", "save_trainable_only", "aggregator_eval", "photometric", "class_weights"):
        setattr(args, k, bool(getattr(args, k)))
    args.no_curriculum = bool(args.no_curriculum)
    args.keep_empty_labels = bool(args.keep_empty_labels)
    # --img-size 可以是 "518" 或 "518x728" (HxW, 都必须是 14 的倍数)
    from aic_data.dataset import _as_hw
    args.img_h, args.img_w = _as_hw(args.img_size)
    # 模型侧的 img_size 只用于初始化 patch_embed 的 pos_embed (之后按实际输入插值)
    args.model_img_size = max(args.img_h, args.img_w)
    # 每个 query 分配一个语义类 (逐 query 监督); 课程阶段会在 set_stage() 里重新分配
    args.assigned_cls = query_class_assignment(args.num_queries, args.num_classes)
    args.stage_class_mask = None
    args.stage_lut = None
    args.stage_classes_now = list(range(1, args.num_classes))
    return args


def main():
    args = parse_args()
    rank, world, local_rank = setup_distributed()
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)

    log(f"[SETUP] epochs={args.epochs} lr={args.lr} batch={args.batch_size} img={args.img_size} "
        f"world={world} freeze_backbone={args.freeze_backbone} num_frames={args.num_frames}")

    # ---- 模型 ----
    model = build_model(args).to(args.device)
    if args.pretrained_ckpt:
        load_checkpoint_into(model, args.pretrained_ckpt, "pretrained(SegStreamVGGT)", allow_missing_query=False)
    else:
        assert os.path.exists(args.backbone_ckpt), f"缺少主干权重 {args.backbone_ckpt}"
        load_checkpoint_into(model, args.backbone_ckpt, "StreamVGGT backbone")
    freeze_backbone(model, args.freeze_backbone, args.train_scope)
    if args.train_scope != "seg" and not args.freeze_backbone:
        log("[MODEL] 注意: --freeze-backbone 0 会解冻全部参数 (含聚合器主干)")

    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    model.train()
    if args.freeze_backbone and args.aggregator_eval:
        unwrap(model).aggregator.eval()

    # ---- 数据 ----
    train_ds, train_loader, val_ds, val_loader, dataset_names = build_loaders(args, rank, world)
    class_weights = compute_class_weights(train_ds, args.class_weight_cap) if args.class_weights else None
    if class_weights is not None:
        log("[LOSS] median-frequency class weights: " +
            " ".join(f"{i}:{float(w):.2f}" for i, w in enumerate(class_weights.tolist())))
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    optimizer = make_optimizer(unwrap(model), args)
    scheduler = make_scheduler(optimizer, args, total_steps)
    scaler = None  # bf16 不需要 GradScaler

    # ---- 分阶段课程 ----
    curriculum = [[args.epochs, list(range(1, args.num_classes)), "不分阶段 (一次监督全部类)"]]
    if not args.no_curriculum:
        curriculum = build_curriculum(args, dataset_names)
    epoch_to_stage, e = {}, 1
    for si, (n_ep, _cls, _lab) in enumerate(curriculum, 1):
        for _ in range(n_ep):
            epoch_to_stage[e] = si
            e += 1
    while e <= args.epochs:                     # epochs 多于课程总和: 都归入最后一段
        epoch_to_stage[e] = len(curriculum)
        e += 1
    log(f"[CURRICULUM] {len(curriculum)} 个阶段:")
    for si, (n_ep, cls, lab) in enumerate(curriculum, 1):
        log(f"[CURRICULUM]   stage {si}: {n_ep} epoch | {len(cls)} 类 {cls} | {lab}")
    cur_stage = 0

    start_epoch = 1
    best_miou = -1.0
    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cpu", weights_only=False)
        load_checkpoint_into(unwrap(model), args.resume, "resume")
        if "optimizer" in sd:
            optimizer.load_state_dict(sd["optimizer"])
        if "scheduler" in sd:
            scheduler.load_state_dict(sd["scheduler"])
        start_epoch = int(sd.get("epoch", 0)) + 1
        best_miou = float(sd.get("val_miou", -1.0))
        log(f"[RESUME] 从 {args.resume} 恢复到 epoch {start_epoch}, best mIoU={best_miou:.4f}")
        if start_epoch > args.epochs:
            log("[RESUME] 已到达 --epochs, 退出")
            return

    tb = None
    if is_main():
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(log_dir=os.path.join(args.log_dir, "tensorboard"))
            log(f"[SETUP] TensorBoard: 已开启 -> {os.path.join(args.log_dir, 'tensorboard')}")
            log(f"[SETUP]   tensorboard --logdir {os.path.join(args.log_dir, 'tensorboard')} --port 6008")
        except ImportError:
            log("[SETUP] 未安装 tensorboard, 跳过")

    metrics_path = os.path.join(args.log_dir, "metrics.csv")
    if is_main() and not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("epoch,stage,lr,train_loss,train_ce,train_dice,val_loss,val_miou,"
                    + ",".join(f"miou_{d}" for d in dataset_names) + "\n")

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        # ---- 阶段切换 ----
        si = epoch_to_stage.get(epoch, len(curriculum))
        if si != cur_stage:
            cur_stage = si
            allowed = set_stage(args, curriculum[si - 1][1])
            log(f"[CURRICULUM] >>> epoch {epoch}: 进入 stage {si}/{len(curriculum)} "
                f"| 监督 {len(allowed)} 类 {allowed} | {curriculum[si - 1][2]}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        train_loader.sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch)
        t0 = time.time()
        rl = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, args, class_weights)
        global_step += len(train_loader)
        log(f"[EPOCH {epoch}] train_loss {rl.mean:.4f} ({time.time()-t0:.0f}s)")
        if tb is not None:
            tb.add_scalar("epoch/train_loss", rl.mean, epoch)
            for k, v in rl.mean_parts().items():
                tb.add_scalar(f"epoch/train_{k}", v, epoch)
            tb.add_scalar("epoch/lr", scheduler.get_last_lr()[0], epoch)

        if epoch % args.val_every == 0 or epoch == args.epochs:
            summary = evaluate(model, val_loader, args, dataset_names, class_weights)
            miou = summary["ALL"]["miou"]
            log(f"[EPOCH {epoch}] (stage {cur_stage}) val_loss {summary['loss']:.4f} | mIoU(all) {miou:.4f}")
            qd = summary.get("query_diag")
            if qd:
                log(f"    [query 诊断] 每样本不同类别数 {qd['unique_classes_per_sample']:.1f}/{args.num_queries}, "
                    f"平均 mask 面积 {qd['mask_area_mean']:.4f}")
            for d in dataset_names:
                dd = summary[d]
                log(f"    {d:11s} mIoU {dd['miou']:.4f}  ({len(dd['iou'])} 类有像素)")
            stage_iou = [summary["ALL"]["iou"][c] for c in args.stage_classes_now if c in summary["ALL"]["iou"]]
            if stage_iou:
                log(f"    [stage {cur_stage}] 本阶段 {len(stage_iou)} 类的平均 IoU "
                    f"{sum(stage_iou)/len(stage_iou):.4f}")
            if tb is not None:
                tb.add_scalar("epoch/val_loss", summary["loss"], epoch)
                tb.add_scalar("epoch/val_miou", miou, epoch)
                for d in dataset_names:
                    tb.add_scalar(f"miou/{d}", summary[d]["miou"], epoch)
                tb.flush()
            if is_main():
                with open(metrics_path, "a") as f:
                    f.write(f"{epoch},{cur_stage},{scheduler.get_last_lr()[0]:.3e},{rl.mean:.4f},"
                            f"{rl.parts.get('ce',0)/max(rl.count,1):.4f},{rl.parts.get('dice',0)/max(rl.count,1):.4f},"
                            f"{summary['loss']:.4f},{miou:.4f},"
                            + ",".join(f"{summary[d]['miou']:.4f}" for d in dataset_names) + "\n")
                with open(os.path.join(args.log_dir, "last_metrics.json"), "w") as f:
                    json.dump({"epoch": epoch, "stage": cur_stage, "lr": scheduler.get_last_lr()[0],
                               "train_loss": rl.mean, "summary": summary}, f, indent=1)
                ckpt = {
                    "epoch": epoch,
                    "stage": cur_stage,
                    "model": (unwrap(model).state_dict() if not args.save_trainable_only else
                              {k: v for k, v in unwrap(model).state_dict().items()
                               if k in unwrap(model).seg_branch_named_parameters()}),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args),
                    "val_miou": miou,
                }
                torch.save(ckpt, os.path.join(args.log_dir, "checkpoint_last.pth"))
                if miou > best_miou:
                    best_miou = miou
                    torch.save(ckpt, os.path.join(args.log_dir, "checkpoint_best.pth"))
                    log(f"[CKPT] 新的最佳 mIoU {miou:.4f} -> checkpoint_best.pth")
        if dist.is_initialized():
            dist.barrier()

    if tb is not None:
        tb.close()
    if dist.is_initialized():
        dist.barrier()
    log(f"[DONE] 训练结束, 最佳 val mIoU = {best_miou:.4f}; 输出目录 {args.log_dir}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*Grad strides do not match bucket view strides.*")
    warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
    main()
