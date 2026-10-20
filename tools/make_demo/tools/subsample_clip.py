#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create a lighter clip from an existing reconstruction cache.

Useful for laptops / software WebGL, or for quickly sharing a demo clip::

    python tools/subsample_clip.py --src data/clips/city2-demo \
        --dst data/clips/city2-lite --stride 5 --label "city2 (lite)"

``stride=5`` keeps about 20% of the points.  Waypoint layout, reveal ranks,
camera poses, route and thumbnails are preserved; only ``points.bin`` and the
per-waypoint counts are rewritten.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    data = core.load_point_buffer(src / "points.bin")
    n = len(data["points"])
    s = max(1, int(args.stride))
    keep = np.arange(0, n, s, dtype=np.int64)
    # always keep the first/last point of each waypoint so every waypoint survives
    wp = data["wp"]
    for w in np.unique(wp):
        idx = np.where(wp == w)[0]
        keep = np.unique(np.r_[keep, idx[0], idx[-1]])
    keep.sort()

    counts, offsets = [], []
    acc = 0
    for w in range(len(meta["waypoints"])):
        sel = keep[wp[keep] == w]
        offsets.append(acc)
        counts.append(int(len(sel)))
        acc += len(sel)

    # rebuild planar buffer
    m = len(keep)
    buf = bytearray(20 * m)
    buf[0:12 * m] = data["points"][keep].astype("<f4").tobytes()
    buf[12 * m:15 * m] = data["rgb"][keep].tobytes()
    buf[15 * m:16 * m] = data["cls"][keep].tobytes()
    buf[16 * m:17 * m] = wp[keep].tobytes()
    buf[17 * m:18 * m] = data["rank"][keep].tobytes()
    buf[18 * m:19 * m] = data["conf"][keep].tobytes()
    buf[19 * m:20 * m] = data["prob"][keep].tobytes()

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "points.bin").write_bytes(bytes(buf))
    if (src / "thumbs").exists():
        if (dst / "thumbs").exists():
            shutil.rmtree(dst / "thumbs")
        shutil.copytree(src / "thumbs", dst / "thumbs")
    meta["id"] = dst.name
    meta["label"] = args.label or (meta.get("label", dst.name) + f" · {s}x 降采样")
    meta["num_points"] = m
    meta["counts"] = counts
    meta["offsets"] = offsets
    meta["subsample_stride"] = s
    meta["class_counts"] = [int((data["cls"][keep] == i).sum()) for i in range(12)]
    for i, wp_obj in enumerate(meta["waypoints"]):
        wp_obj["count"] = counts[i]
        wp_obj["offset"] = offsets[i]
    (dst / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SUBSAMPLE] {src} -> {dst}: {n:,} -> {m:,} points (stride {s})")


if __name__ == "__main__":
    main()
