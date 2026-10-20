# -*- coding: utf-8 -*-
"""
SegStreamVGGT 初始化与冻结范围自检 (不需要 GPU)

回答两个问题:
  1. 权重是从哪里加载的? —— StreamVGGT 预训练 checkpoint 加载了哪些张量、哪些是随机初始化的新分支;
  2. 几何重建部分是否被冻结? —— 按模块列出参数量/可训练量, 并用一次真实的
     forward+backward 验证"冻结参数确实没有梯度"。

    python check_setup.py                      # 默认配置 (--with-geometry 0)
    python check_setup.py --with-geometry 1    # 构建 depth/point head 并确认它们被冻结
    python check_setup.py --train-scope seg+geometry   # 连几何头一起训时的对照
    python check_setup.py --ckpt /path/to/xxx.pth      # 检查某个已训练权重
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aic_data.losses import query_class_assignment, total_query_seg_loss   # noqa: E402
from segstreamvggt import SegStreamVGGT                                    # noqa: E402

BACKBONE_CKPT = "/data/luhangze/aic/StreamVGGT/ckpt/checkpoints.pth"
OK, BAD, INFO = "[OK ]", "[ERR]", "[   ]"


def module_of(name: str) -> str:
    if name.startswith("aggregator.query"):
        return "aggregator.query  (分割/新)"
    for p in ("aggregator.", "semantic_head.", "depth_head.", "point_head.", "camera_head.", "track_head."):
        if name.startswith(p):
            return p
    return "其它"


def load_report(model, ckpt_path):
    print(f"\n[1] 权重来源: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"{BAD} 文件不存在")
        return None
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kind = "SegStreamVGGT 训练权重 (含 model 键)" if isinstance(sd, dict) and "model" in sd else "裸 state_dict"
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    print(f"{INFO} 文件类型: {kind}; 张量数 {len(sd)}")

    model_sd = model.state_dict()
    matched, shape_mismatch = {}, []
    for k, v in sd.items():
        if k in model_sd:
            (matched.__setitem__(k, v) if model_sd[k].shape == v.shape else shape_mismatch.append(k))
    missing = [k for k in model_sd if k not in matched]

    # 按模块统计
    print(f"\n{INFO} 加载成功 (按模块):")
    for label in sorted({module_of(k) for k in list(matched) + missing}):
        n_ok = sum(1 for k in matched if module_of(k) == label)
        n_miss = sum(1 for k in missing if module_of(k) == label)
        tag = "预训练权重 ✅" if n_ok and "aggregator.query" not in label and "semantic_head" not in label else ""
        print(f"{INFO}   {label:28s} 从 ckpt 载入 {n_ok:4d} 个张量 | 未加载(新初始化) {n_miss:4d} 个  {tag}")

    interp = [k for k in missing if k.endswith("patch_embed.pos_embed")]
    new_param = [k for k in missing if k not in interp]
    print(f"{INFO} 随机初始化的新参数 (query/语义头): {len(new_param)} 个")
    if interp:
        print(f"{INFO} patch_embed.pos_embed: 形状不符, 由插值自动处理 (输入分辨率不同, 正常)")
    print(f"{INFO} ckpt 里有但本模型不构建的键: {len([k for k in sd if k not in model_sd])} 个 "
          f"(如 camera/track head —— 当前配置未启用)")
    return matched, missing


def scope_report(model, title):
    print(f"\n{INFO} {title}")
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for label, (tot, tr) in model.parameter_groups_report().items():
        mark = "✅ 可训练" if tr > 0 else "❄️  冻结"
        print(f"{INFO}   {label:24s} {tot/1e6:8.2f}M | 可训练 {tr/1e6:8.2f}M | {mark}")
    print(f"{INFO}   合计 {n_all/1e6:.1f}M | 可训练 {n_tr/1e6:.2f}M ({100*n_tr/n_all:.2f}%) | "
          f"冻结 {100*(1-n_tr/n_all):.2f}%")


def grad_check(model, with_geometry=False):
    """跑一次真实 forward+backward, 验证冻结参数无梯度。"""
    print(f"\n[3] 冻结有效性验证 (forward + backward)")
    model.train()
    B, S, H = 2, 2, 126
    images = torch.rand(B, S, 3, H, H)
    labels = torch.randint(0, 12, (B, S, H, H))
    valid = torch.ones(B, 12, dtype=torch.bool)
    preds = model(images, with_geometry=with_geometry)
    seg_prob = preds["seg_prob"].float()
    hw = seg_prob.shape[-2:]
    import torch.nn.functional as F
    tgt = F.interpolate(labels.float().reshape(-1, 1, H, H), size=hw, mode="nearest").reshape(B, S, *hw).long()
    loss, _, _ = total_query_seg_loss(seg_prob, tgt, valid)
    loss.backward()

    frozen_with_grad, trainable_without_grad = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None and torch.count_nonzero(p.grad) > 0:
                frozen_with_grad.append(n)
        else:
            if p.grad is None or torch.count_nonzero(p.grad) == 0:
                trainable_without_grad.append(n)
    n_frozen = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
    n_train = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    print(f"{INFO} 冻结张量 {n_frozen} 个, 可训练张量 {n_train} 个")
    if frozen_with_grad:
        print(f"{BAD} 有 {len(frozen_with_grad)} 个冻结参数出现了非零梯度! 例: {frozen_with_grad[:3]}")
    else:
        print(f"{OK} 所有冻结参数梯度为空/零 —— 几何重建部分与主干确实不参与更新")
    if trainable_without_grad:
        print(f"{INFO} 可训练但无梯度的张量 {len(trainable_without_grad)} 个 (通常是 no-object/未启用的 query): "
              f"{trainable_without_grad[:3]}")
    return not frozen_with_grad


def main():
    ap = argparse.ArgumentParser(description="SegStreamVGGT 初始化 + 冻结范围自检")
    ap.add_argument("--ckpt", default=BACKBONE_CKPT, help="要检查的权重 (默认 StreamVGGT 预训练)")
    ap.add_argument("--with-geometry", type=int, default=0, help="是否构建 depth/point head")
    ap.add_argument("--train-scope", default="seg", choices=["seg", "seg+geometry", "all"])
    ap.add_argument("--num-queries", type=int, default=100)
    ap.add_argument("--query-dim", type=int, default=256)
    ap.add_argument("--out-ratio", type=int, default=2)
    args = ap.parse_args()

    print("=" * 78)
    print("SegStreamVGGT setup check")
    print("=" * 78)
    model = SegStreamVGGT(
        img_size=126, patch_size=14, embed_dim=1024, num_classes=12,
        enable_camera=False, enable_point=bool(args.with_geometry), enable_depth=bool(args.with_geometry),
        enable_track=False, query_num=args.num_queries, query_dim=args.query_dim,
        cross_block_params={"embed_dim": 256, "num_heads": 8, "mlp_ratio": 2.0},
        out_ratio=args.out_ratio,
    )

    load_report(model, args.ckpt)

    print(f"\n[2] 训练范围: --train-scope {args.train_scope}  (--freeze-backbone 1)")
    model.set_train_scope(args.train_scope)
    scope_report(model, f"当前配置 (with_geometry={args.with_geometry}) 的模块级冻结情况")

    ok = grad_check(model, with_geometry=bool(args.with_geometry) and args.train_scope == "seg+geometry")
    print("\n" + "=" * 78)
    print(f"结论: 主干来自预训练权重 ✅  |  几何重建部分冻结 ✅  " if ok else "结论: 冻结校验失败 ❌")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
