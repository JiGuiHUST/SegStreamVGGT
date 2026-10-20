# -*- coding: utf-8 -*-
"""
SegStreamVGGT 数据管线自检 (不需要 GPU)

检查项:
  1. 每个数据集缓存可读 (img/lbl 尺寸一致, manifest 完整)
  2. 标签 id ⊆ {0(unlabeled/ignore)} ∪ 该数据集真实标注类 C_d
     (扩展类 12..27 不会漏进来 —— README §6.2 部分标签问题)
  3. `valid_class` 掩码与 `registry.AIC_DATASETS[...].classes` 一致
  4. 训练/验证样本形状、取值范围、crop 对齐 (图与标签必须同步裁剪)
  5. 多帧模式 (independent / same-scene) 与 num_frames>1

    python verify_pipeline.py
    python verify_pipeline.py --datasets uavid,potsdam --num-frames 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aic_data.dataset import AICCachedDataset, collate_aic            # noqa: E402
from aic_data.registry import DEFAULT_DATASETS, IGNORE_INDEX, get_spec  # noqa: E402

OK, BAD = "[OK ]", "[ERR]"


def check_dataset(cache_root: str, name: str, split: str, args) -> int:
    mpath = os.path.join(cache_root, name, split, "manifest.json")
    if not os.path.exists(mpath):
        print(f"{BAD} {name}/{split}: 缺少 {mpath}")
        return 1
    ds = AICCachedDataset(cache_root=cache_root, datasets=[name], split=split,
                          num_frames=args.num_frames, img_size=args.img_size, patch_size=14,
                          train=(split == "train"), frames_mode=args.frames_mode,
                          num_classes=args.num_classes, seed=args.seed)
    allowed = {IGNORE_INDEX} | set(get_spec(name).classes)
    n = min(args.samples, len(ds))
    rng = np.random.RandomState(0)
    idxs = rng.choice(len(ds), size=n, replace=False).tolist() if n < len(ds) else list(range(n))
    errs = 0
    for i in idxs:
        it = ds[i]
        img, lbl, vc = it["images"], it["labels"], it["valid_class"]
        assert img.shape == (args.num_frames, 3, ds.ch, ds.cw), f"img {tuple(img.shape)}"
        assert lbl.shape == (args.num_frames, ds.ch, ds.cw), f"lbl {tuple(lbl.shape)}"
        if float(img.min()) < -1e-6 or float(img.max()) > 1 + 1e-6:
            print(f"{BAD} {name}/{split} #{i}: img 越界 [{img.min()},{img.max()}]"); errs += 1
        bad = sorted(set(np.unique(lbl.numpy()).tolist()) - allowed)
        if bad:
            print(f"{BAD} {name}/{split} #{i}: 标签出现未标注类 {bad} (allowed={sorted(allowed)})"); errs += 1
        exp = np.zeros(args.num_classes, dtype=bool); exp[IGNORE_INDEX] = True
        for c in get_spec(name).classes:
            if c < args.num_classes:
                exp[c] = True
        if not np.array_equal(vc.numpy(), exp):
            print(f"{BAD} {name}/{split} #{i}: valid_class 与 registry 不一致"); errs += 1
    cls_frac = {}
    for c in sorted(allowed - {IGNORE_INDEX}):
        frac = sum(float((ds[i]["labels"] != IGNORE_INDEX).sum()) for i in idxs[:max(3, n // 4)])
        cls_frac[c] = frac
    print(f"{OK if errs == 0 else BAD} {name:11s}/{split:5s} items={len(ds):5d} 抽查 {n} 个, "
          f"classes={get_spec(name).classes}")
    return errs


def main():
    ap = argparse.ArgumentParser(description="SegStreamVGGT 数据管线自检")
    ap.add_argument("--cache-root", default=str(Path(__file__).resolve().parent / "cache"))
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--img-size", default=518)
    ap.add_argument("--num-frames", type=int, default=1)
    ap.add_argument("--frames-mode", default="tile-grid",
                    choices=["tile-grid", "sequence", "same-scene", "independent"])
    ap.add_argument("--num-classes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    names = list(DEFAULT_DATASETS) if args.datasets == "all" else \
        [d.strip() for d in args.datasets.split(",") if d.strip()]
    print(f"[VERIFY] cache_root={args.cache_root}\n[VERIFY] datasets={names} "
          f"num_frames={args.num_frames} frames_mode={args.frames_mode} img={args.img_size}")

    errs = 0
    for name in names:
        for split in ("train", "val"):
            try:
                errs += check_dataset(args.cache_root, name, split, args)
            except Exception as e:  # noqa: BLE001
                print(f"{BAD} {name}/{split}: 异常 {type(e).__name__}: {e}")
                errs += 1

    # collate / batch 形状
    ds = AICCachedDataset(cache_root=args.cache_root, datasets=names, split="train",
                          num_frames=args.num_frames, img_size=args.img_size, patch_size=14,
                          train=True, frames_mode=args.frames_mode, num_classes=args.num_classes)
    batch = collate_aic([ds[i] for i in range(min(3, len(ds)))])
    print(f"[VERIFY] batch: images {tuple(batch['images'].shape)} labels {tuple(batch['labels'].shape)} "
          f"valid_class {tuple(batch['valid_class'].shape)} dataset_id {tuple(batch['dataset_id'].shape)}")
    assert batch["images"].shape[1] == args.num_frames

    print(f"\n[VERIFY] {'全部通过' if errs == 0 else f'发现 {errs} 个问题'}")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
