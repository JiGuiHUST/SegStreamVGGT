# -*- coding: utf-8 -*-
"""
快速抽样核对 (每个数据集抽几张, 出图人工看)

对每个数据集抽 train 2 张 + val 1 张, 渲染三联对比图:
    [RGB]  [GT 伪彩]  [RGB+GT 叠加]      -> spotcheck/<dataset>.jpg
同时打印每帧的尺寸 / 像素占比前 4 类 / 标签-影像对齐得分 (降采样加速)。

    python spot_check.py                    # 7 个数据集
    python spot_check.py --datasets dstl,uavid --per-split 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))            # SegStreamVGGT/
sys.path.insert(0, str(THIS.parents[1]))        # aic/ (label_mapping.py)
from aic_data.registry import AIC_DATASETS, IGNORE_INDEX, NUM_AIC_CLASSES   # noqa: E402
import label_mapping as lm                                                  # noqa: E402

CACHE = str(THIS.parent / "cache")
OUT = str(THIS.parent / "spotcheck")
PAL = {i: c for i, _n, c, _d in lm.UNIFIED_CLASSES}
NAME = {i: n for i, n, _c, _d in lm.UNIFIED_CLASSES}


def load(ds, split, name):
    base = os.path.join(CACHE, ds, split)
    img = np.asarray(Image.open(os.path.join(base, "img", name + ".jpg")).convert("RGB"))
    lbl = np.asarray(Image.open(os.path.join(base, "lbl", name + ".png")))
    return img, lbl


def colorize(lbl):
    out = np.zeros(lbl.shape + (3,), np.uint8)
    for k, c in PAL.items():
        out[lbl == k] = c
    return out


def overlay(img, lbl, alpha=0.55):
    col = colorize(lbl)
    m = lbl != IGNORE_INDEX
    out = img.copy()
    out[m] = (1 - alpha) * img[m] + alpha * col[m]
    return out.astype(np.uint8)


def align_score(img, lbl, factor=4):
    """标签边界处影像梯度 / 全图梯度 (整数降采样加速; 对齐 >1.3 视为良好)。"""
    h, w = lbl.shape
    f = max(1, min(factor, min(h, w) // 256))
    small_img = Image.fromarray(img).resize((max(1, w // f), max(1, h // f)), Image.BILINEAR)
    small_lbl = np.asarray(Image.fromarray(lbl).resize((max(1, w // f), max(1, h // f)), Image.NEAREST))
    g = np.asarray(small_img.convert("L"), np.float32)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    b = np.zeros_like(small_lbl, bool)
    b[1:, :] |= small_lbl[1:, :] != small_lbl[:-1, :]
    b[:, 1:] |= small_lbl[:, 1:] != small_lbl[:, :-1]
    b &= small_lbl != IGNORE_INDEX
    if b.sum() < 20:
        return float("nan")
    return float(mag[b].mean() / max(mag.mean(), 1e-6))


def fit_width(a, w=420):
    h = a.shape[0]
    if a.shape[1] == w:
        return a
    return np.asarray(Image.fromarray(a).resize((w, max(1, int(h * w / a.shape[1]))), Image.BILINEAR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--per-split", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    ds_list = list(AIC_DATASETS) if args.datasets == "all" else [d.strip() for d in args.datasets.split(",")]
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"{'dataset':11s} {'split':5s} {'name':34s} {'尺寸':>13s} {'对齐':>6s}  像素占比 (前4)")
    print("-" * 118)
    for ds in ds_list:
        rows = []
        for split in ("train", "val"):
            mpath = os.path.join(CACHE, ds, split, "manifest.json")
            if not os.path.exists(mpath):
                continue
            items = json.load(open(mpath))["items"]
            if not items:
                continue
            # 优先挑 valid_px 大的 (内容丰富的), 但也展示一张小的
            items = sorted(items, key=lambda it: -it.get("valid_px", 0))
            pool = items[: max(args.per_split * 8, 12)]
            pick = rng.sample(pool, min(args.per_split, len(pool)))
            if items and items[-1]["name"] not in [p["name"] for p in pick]:
                pick.append(items[-1])          # 再加一张"最空"的, 便于发现问题
            for it in pick:
                img, lbl = load(ds, split, it["name"])
                sc = align_score(img, lbl)
                cnt = {int(k): int(v) for k, v in zip(*np.unique(lbl, return_counts=True))}
                tot = sum(v for k, v in cnt.items() if k != IGNORE_INDEX) or 1
                top = sorted(((k, v) for k, v in cnt.items() if k != IGNORE_INDEX),
                             key=lambda kv: -kv[1])[:4]
                frac = ", ".join(f"{NAME[k]} {100*v/tot:.0f}%" for k, v in top) or "(全 ignore)"
                rows.append((split, it["name"], img, lbl))
                print(f"{ds:11s} {split:5s} {it['name'][:34]:34s} "
                      f"{img.shape[1]:5d}x{img.shape[0]:<5d} {sc:6.2f}  {frac}")
        if not rows:
            continue
        tiles = []
        for split, name, img, lbl in rows:
            a = fit_width(img)
            b = fit_width(colorize(lbl))
            c = fit_width(overlay(img, lbl))
            h = min(x.shape[0] for x in (a, b, c))
            tiles.append(np.concatenate([a[:h], b[:h], c[:h]], axis=1))
        H = min(t.shape[0] for t in tiles)
        W = tiles[0].shape[1]
        sheet = np.concatenate([t[:H, :W] for t in tiles], axis=0)
        p = os.path.join(OUT, f"{ds}.jpg")
        Image.fromarray(sheet).save(p, quality=90)
        print(f"{'':11s} -> 对比图: {p}  ({sheet.shape[1]}x{sheet.shape[0]}, 行=样本, 列=RGB|GT|叠加)")
    print(f"\n全部对比图在 {OUT}/ —— 请直接看图确认标签与影像是否吻合")


if __name__ == "__main__":
    main()
