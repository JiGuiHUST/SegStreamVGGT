# -*- coding: utf-8 -*-
"""
SegStreamVGGT: 在 AIC 验证集上评测 + 可视化

用法:
    # 评测训练好的 checkpoint (逐数据集 mIoU + 逐类 IoU + 可视化)
    python eval_aic.py --ckpt logs_aic/checkpoint_best.pth

    # 同时评测未训练基线 (随机 query 分支), 便于对比训练收益
    python eval_aic.py --ckpt logs_aic/checkpoint_best.pth --baseline

    # 指定数据集 / 样本量 / 输出目录
    python eval_aic.py --ckpt logs_aic/checkpoint_best.pth \
        --datasets uavid,potsdam,hdmaps --max-val-samples 200 --num-viz 8

输出 (--output-dir, 默认 <ckpt 所在目录>/eval_aic):
    metrics.json          逐数据集/逐类 IoU, loss
    metrics_table.txt     便读表格
    viz/frame_XX.png      每张: RGB | GT(伪彩) | 预测(伪彩) | 置信度
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aic_data.dataset import AICCachedDataset, collate_aic                       # noqa: E402
from aic_data.losses import IoUMeter, RunningLoss, total_query_seg_loss          # noqa: E402
from aic_data.registry import (AIC_NAME, DEFAULT_DATASETS, IGNORE_INDEX, get_spec)  # noqa: E402
from train_aic import build_model, load_checkpoint_into                          # noqa: E402

PALETTE = {
    0: (0, 0, 0), 1: (128, 0, 0), 2: (128, 64, 128), 3: (192, 192, 192), 4: (128, 128, 64),
    5: (0, 128, 0), 6: (128, 128, 0), 7: (192, 128, 64), 8: (0, 0, 192), 9: (192, 0, 192),
    10: (64, 64, 0), 11: (0, 192, 255),
}


def colorize(idx: np.ndarray) -> np.ndarray:
    out = np.zeros(idx.shape + (3,), dtype=np.uint8)
    for k, c in PALETTE.items():
        out[idx == k] = c
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="SegStreamVGGT AIC 评测")
    ap.add_argument("--ckpt", required=True, help="checkpoint_best.pth / checkpoint_last.pth / 纯 state_dict")
    ap.add_argument("--backbone-ckpt", default="/data/luhangze/aic/StreamVGGT/ckpt/checkpoints.pth",
                    help="仅在 --baseline 或 ckpt 缺主干时使用")
    ap.add_argument("--baseline", type=int, default=0, help="额外评测未训练基线 (随机 query 分支)")
    ap.add_argument("--datasets", default="all", help="all 或逗号分隔, 如 uavid,potsdam")
    ap.add_argument("--cache-root", default=str(Path(__file__).resolve().parent / "cache"))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--img-size", default=518, help="输入尺寸: 518 或 518x728 (HxW)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-val-samples", type=int, default=400, help="每个数据集的分层验证样本数上限 (0=全部)")
    ap.add_argument("--val-crops", type=int, default=4)
    ap.add_argument("--full-image", type=int, default=0,
                    help="=1 时忽略 val-crops/max-val-samples, 用全部不重叠 518 crop 覆盖整张图 (推荐出正式数字)")
    ap.add_argument("--frames-mode", default="tile-grid")
    ap.add_argument("--val-num-frames", type=int, default=1, help="验证时每样本帧数 (1 = 逐 crop 评测)")
    ap.add_argument("--num-classes", type=int, default=12)
    ap.add_argument("--num-queries", type=int, default=100)
    ap.add_argument("--query-dim", type=int, default=256)
    ap.add_argument("--cross-dim", type=int, default=256)
    ap.add_argument("--cross-heads", type=int, default=8)
    ap.add_argument("--out-ratio", type=int, default=2,
                    help="必须与训练一致 (v4/旧 best 均为 2; 旧默认 4 会导致特征图更粗)")
    ap.add_argument("--with-geometry", type=int, default=0)
    ap.add_argument("--query-self-attention", type=int, default=0)
    ap.add_argument("--cross-block-layers", default=None)
    ap.add_argument("--supervise-attend-frame", type=int, default=0)
    ap.add_argument("--query-cls-hidden-dim", type=int, default=64)
    ap.add_argument("--num-viz", type=int, default=6, help="可视化样本数 (0=关闭)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.cross_block_layers:
        args.cross_block_layers = [int(x) for x in str(args.cross_block_layers).split(",")]
    for k in ("baseline", "with_geometry", "query_self_attention", "supervise_attend_frame", "full_image"):
        setattr(args, k, bool(getattr(args, k)))
    if args.full_image:
        args.val_crops = 0          # 0 = 所有不重叠 crop
        args.max_val_samples = 0    # 0 = 全部
    from aic_data.dataset import _as_hw
    args.img_h, args.img_w = _as_hw(args.img_size)
    args.model_img_size = max(args.img_h, args.img_w)
    return args


@torch.no_grad()
def evaluate_model(model, loader, args, dataset_names, viz_dir=None, num_viz=0):
    model.eval()
    meter = IoUMeter(args.num_classes, dataset_names, ignore_index=IGNORE_INDEX)
    rl = RunningLoss()
    viz_done = 0
    # query 分支诊断 (防止 query 塌缩: 所有 query 都预测同一类/同一片区域)
    diag = {"n_unique_cls": [], "mask_mean": [], "mask_max": [], "samples": 0}
    for batch in loader:
        images = batch["images"].cuda(non_blocking=True)
        labels = batch["labels"]
        valid_class = batch["valid_class"].cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = model(images, with_geometry=args.with_geometry)
        seg_prob = preds["seg_prob"].float()                     # B,C,S,h,w

        # 诊断: query 分类多样性与 mask 面积
        if "query_logits" in preds:
            qcls = preds["query_logits"].float().argmax(dim=-1)             # B,Q
            qmask = torch.sigmoid(preds["query_masks"].float())             # B,Q,S,h,w
            diag["n_unique_cls"].append(
                float(np.mean([len(torch.unique(qcls[b])) for b in range(qcls.shape[0])])))
            per_q_area = qmask.mean(dim=(2, 3, 4))                          # B,Q
            diag["mask_mean"].append(float(per_q_area.mean()))
            diag["mask_max"].append(float(per_q_area.max()))
            diag["samples"] += int(qcls.shape[0])

        # 验证 loss (与训练同口径: 在特征图上计算)
        target = F.interpolate(
            labels.float().reshape(-1, 1, *labels.shape[-2:]).cuda(), size=seg_prob.shape[-2:], mode="nearest"
        ).reshape(labels.shape[0], labels.shape[1], *seg_prob.shape[-2:]).long()
        loss, ce, dice = total_query_seg_loss(seg_prob, target, valid_class, ignore_index=IGNORE_INDEX)
        rl.update(loss.item(), ce=float(ce), dice=float(dice))

        B, C, S, h, w = seg_prob.shape
        for s in range(S):
            up = F.interpolate(seg_prob[:, :, s], size=labels.shape[-2:], mode="bilinear", align_corners=False)
            up = up * valid_class[:, :, None, None].to(up.dtype)
            if up.shape[1] > 1:
                up[:, 0] = -1.0          # class 0 = ignore/void, 不作为预测
            conf, pred = up.max(dim=1)
            meter.update(pred.cpu(), labels[:, s], batch["dataset_id"], valid_class.cpu())

            if viz_dir is not None and viz_done < num_viz:
                for b in range(B):
                    if viz_done >= num_viz:
                        break
                    rgb = (images[b, s].float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    gt = colorize(labels[b, s].numpy().astype(np.uint8))
                    pd = colorize(pred[b].cpu().numpy().astype(np.uint8))
                    cf = (conf[b].float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    cf = np.stack([cf] * 3, axis=-1)
                    row = np.concatenate([rgb, gt, pd, cf], axis=1)
                    Image.fromarray(row).save(os.path.join(
                        viz_dir, f"frame_{viz_done:02d}_{batch['dataset_name'][b]}_{batch['names'][b][0].split('/')[-1][:32]}.png"))
                    viz_done += 1

    valid_classes = {name: list(get_spec(name).classes) for name in dataset_names}
    summary = meter.summary(valid_classes)
    summary["loss"] = rl.mean
    summary["parts"] = rl.mean_parts()
    if diag["samples"]:
        summary["query_diag"] = {
            "unique_classes_per_sample": float(np.mean(diag["n_unique_cls"])),
            "mask_area_mean": float(np.mean(diag["mask_mean"])),
            "mask_area_max": float(np.mean(diag["mask_max"])),
        }
    return summary


def write_table(summary, dataset_names, out_path):
    lines = []
    header = f"{'dataset':<12} {'mIoU':>7} " + " ".join(f"{AIC_NAME[c][:6]:>7}" for c in range(1, 12))
    lines.append(header)
    lines.append("-" * len(header))
    for name in dataset_names + ["ALL"]:
        s = summary.get(name)
        if not s:
            continue
        row = f"{name:<12} {s['miou']:>7.4f} "
        row += " ".join(f"{s['iou'].get(c, float('nan')):>7.3f}" if c in s["iou"] else f"{'-':>7}"
                        for c in range(1, 12))
        lines.append(row)
    lines.append("")
    lines.append(f"loss={summary.get('loss', float('nan')):.4f}  parts={summary.get('parts')}")
    txt = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(txt + "\n")
    return txt


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dataset_names = list(DEFAULT_DATASETS) if args.datasets == "all" else \
        [d.strip() for d in args.datasets.split(",") if d.strip()]
    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "eval_aic")
    os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(out_dir, "viz")
    if args.num_viz > 0:
        os.makedirs(viz_dir, exist_ok=True)
    print(f"[EVAL] ckpt={args.ckpt}\n[EVAL] datasets={dataset_names}\n[EVAL] output={out_dir}", flush=True)

    val_ds = AICCachedDataset(
        cache_root=args.cache_root, datasets=dataset_names, split="val",
        num_frames=args.val_num_frames, img_size=args.img_size, patch_size=14, train=False,
        frames_mode=args.frames_mode, val_crops=args.val_crops, num_classes=args.num_classes, seed=args.seed,
    )
    # 每个数据集各自分层截断 (默认 400): 等间隔取样, 小数据集全部评到
    by_ds = {}
    for i, s in enumerate(val_ds.samples):
        by_ds.setdefault(s.dataset, []).append(i)
    idx = []
    for name in dataset_names:
        ids = by_ds.get(name, [])
        if not ids:
            continue
        if args.max_val_samples > 0 and len(ids) > args.max_val_samples:
            step = len(ids) / args.max_val_samples
            ids = [ids[min(len(ids) - 1, int(i * step))] for i in range(args.max_val_samples)]
        idx.extend(ids)
    from torch.utils.data import DataLoader, Subset
    loader = DataLoader(Subset(val_ds, idx), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_aic, pin_memory=True)
    print(f"[EVAL] 验证样本 {len(idx)} (来自 {len(val_ds)} 个, {len(dataset_names)} 个数据集), batch={args.batch_size}",
          flush=True)

    results = {}
    if args.baseline:
        base = build_model(args).to(args.device)
        load_checkpoint_into(base, args.backbone_ckpt, "baseline backbone")
        base.eval()
        print("[EVAL] ---- 基线 (未训练 query 分支) ----", flush=True)
        results["baseline"] = evaluate_model(base, loader, args, dataset_names)
        print(write_table(results["baseline"], dataset_names, os.path.join(out_dir, "metrics_table_baseline.txt")), flush=True)
        del base
        torch.cuda.empty_cache()

    model = build_model(args).to(args.device)
    load_checkpoint_into(model, args.ckpt, "trained SegStreamVGGT", allow_missing_query=False)
    model.eval()
    print("[EVAL] ---- 训练后 ----", flush=True)
    results["trained"] = evaluate_model(model, loader, args, dataset_names,
                                        viz_dir=viz_dir if args.num_viz > 0 else None, num_viz=args.num_viz)
    txt = write_table(results["trained"], dataset_names, os.path.join(out_dir, "metrics_table.txt"))
    print(txt, flush=True)
    qd = results["trained"].get("query_diag")
    if qd:
        print(f"[EVAL] query 诊断: 每样本不同类别数 {qd['unique_classes_per_sample']:.2f}/{args.num_queries} 个 query, "
              f"平均 mask 面积 {qd['mask_area_mean']:.4f}, 最大 query mask 面积 {qd['mask_area_max']:.4f}",
              flush=True)

    if args.baseline:
        print("\n[EVAL] ---- 训练前后对比 (mIoU) ----", flush=True)
        for name in dataset_names + ["ALL"]:
            b = results["baseline"].get(name, {}).get("miou", float("nan"))
            t = results["trained"].get(name, {}).get("miou", float("nan"))
            if b != b and t != t:
                continue
            print(f"  {name:<12} {b:.4f} -> {t:.4f}  ({t - b:+.4f})", flush=True)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=1, default=str)
    print(f"[EVAL] 结果已写入 {out_dir}", flush=True)


if __name__ == "__main__":
    main()
