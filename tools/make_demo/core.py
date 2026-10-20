#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared reconstruction core for the interactive UAV city demo.

This is a refactor of ``SegStreamVGGT/demo/demo_uavid_route.py``:

* the expensive part (model forward + depth/pose unprojection + semantic head)
  is kept exactly as in the original script;
* the old matplotlib / ffmpeg rendering block is removed, and the results are
  written as a compact binary buffer that the browser can consume directly.

World coordinate convention: VGGT world is y-down, while Three.js is Y-up.
So for the web app we use ``(x, -y, z)``.  Everything sent to the browser
(``points``, ``cams``, ``route``, ``bounds``) is already in web coords.

Point buffer layout (little-endian, planar, ``N`` points)::

    offset 0      : float32 positions  3*N   (display x, y, z)
    offset 12*N   : uint8   rgb        3*N
    offset 15*N   : uint8   cls          N
    offset 16*N   : uint8   wp           N    waypoint index
    offset 17*N   : uint8   rank         N    near-to-far reveal rank 0..255
    offset 18*N   : uint8   conf         N    per-frame depth-conf percentile
    offset 19*N   : uint8   prob         N    semantic max-prob * 255

Total 20 bytes / point.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Repo / checkpoints
# ---------------------------------------------------------------------------
DEMO_DIR = Path(__file__).resolve().parent
SEGSTREAM_ROOT = Path(os.environ.get("SEGSTREAM_ROOT", "/data/luhangze/aic/SegStreamVGGT")).resolve()
if str(SEGSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(SEGSTREAM_ROOT))

DEFAULT_CKPT = str(SEGSTREAM_ROOT / "logs_aic_v5/checkpoint_best.pth")
DEFAULT_BACKBONE = "/data/luhangze/aic/StreamVGGT/ckpt/checkpoints.pth"

# AIC-11 palette (identical to eval_aic.py / the original demo)
PALETTE = {
    0: (0, 0, 0), 1: (128, 0, 0), 2: (128, 64, 128), 3: (192, 192, 192),
    4: (128, 128, 64), 5: (0, 128, 0), 6: (128, 128, 0), 7: (192, 128, 64),
    8: (0, 0, 192), 9: (192, 0, 192), 10: (64, 64, 0), 11: (0, 192, 255),
}
NAMES = {0: "unlabeled", 1: "building", 2: "road", 3: "sidewalk", 4: "parking",
         5: "tree", 6: "low_veg", 7: "terrain", 8: "water", 9: "vehicle",
         10: "person", 11: "sky"}
CLASS_NAMES = [NAMES[i] for i in range(12)]
CLASS_LABELS_ZH = ["未标注", "建筑", "道路", "人行道", "停车场", "树木",
                   "低矮植被", "地形", "水面", "车辆", "行人", "天空"]
PALETTE_LIST = [list(PALETTE[i]) for i in range(12)]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_hw(s) -> tuple[int, int]:
    """Parse ``HxW`` or a single square size."""
    if isinstance(s, (list, tuple)):
        h, w = s
        return int(h), int(w)
    if "x" in str(s).lower():
        h, w = str(s).lower().split("x")
        return int(h), int(w)
    return int(s), int(s)


def display_coords(v: np.ndarray) -> np.ndarray:
    """VGGT world (x, y, z) -> web coords (x, -y, z), Y-up (Three.js)."""
    v = np.asarray(v)
    return np.stack([v[..., 0], -v[..., 1], v[..., 2]], axis=-1)


def build_model(ckpt: str = DEFAULT_CKPT, backbone: str = DEFAULT_BACKBONE,
                out_ratio: int = 2, device: str = "cuda:0",
                point_source: str = "depth"):
    """Build the model and load StreamVGGT geometry + SegStreamVGGT seg weights."""
    from segstreamvggt.models.segstreamvggt import SegStreamVGGT
    from train_aic import load_checkpoint_into

    model = SegStreamVGGT(
        img_size=518, patch_size=14, embed_dim=1024, num_classes=12,
        enable_camera=True,
        enable_point=(point_source == "point"),
        enable_depth=(point_source == "depth"),
        enable_track=False,
        query_num=100, query_dim=256,
        cross_block_params={"embed_dim": 256, "num_heads": 8, "mlp_ratio": 2.0},
        out_ratio=out_ratio,
    )
    # 1) StreamVGGT: aggregator + camera / depth(or point) heads
    load_checkpoint_into(model, backbone, "StreamVGGT backbone")
    # 2) SegStreamVGGT: query + semantic head (geometry stays frozen)
    load_checkpoint_into(model, ckpt, "SegStreamVGGT seg")
    model.to(device).eval()
    return model


def list_images(path) -> list[Path]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(str(path))
    files = [q for q in sorted(p.iterdir()) if q.is_file() and q.suffix.lower() in IMAGE_EXTS]
    return files


def select_evenly(items, n: int):
    """Pick ``n`` items evenly spread over ``items`` (keeps order)."""
    items = list(items)
    n = int(n or 0)
    if n <= 0 or len(items) <= n:
        return items
    if n == 1:
        return [items[0]]
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    seen, out = set(), []
    for i in idx:
        if int(i) not in seen:
            seen.add(int(i))
            out.append(items[int(i)])
    return out


def load_frame_files(paths, H: int, W: int):
    """Load RGB frames and resize to (H, W). Returns tensor [1,S,3,H,W], rgbs, names."""
    import torch

    names, rgbs = [], []
    for p in paths:
        p = Path(p)
        im = Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR)
        rgbs.append(np.asarray(im))
        names.append(p.stem)
    if not rgbs:
        raise RuntimeError("没有可用的输入帧")
    x = torch.from_numpy(np.stack(rgbs).astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    return x.unsqueeze(0), rgbs, names


def infer(model, images, H: int, W: int, point_source: str = "depth"):
    """Run the frozen geometry heads + trained seg head on all frames at once."""
    import torch
    import torch.nn.functional as F
    from segstreamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(images, with_geometry=True)

    seg = out["seg_prob"][0].float()                             # C,S,h,w
    C, S, h, w = seg.shape
    seg_up = F.interpolate(seg.permute(1, 0, 2, 3), size=(H, W), mode="bilinear",
                           align_corners=False)                  # S,C,H,W
    prob, pred = seg_up.max(dim=1)                               # S,H,W
    pred = pred.cpu().numpy()
    prob = prob.cpu().numpy()

    pose = out["pose_enc"].float()
    ext, intr = pose_encoding_to_extri_intri(pose, image_size_hw=(H, W))
    R = ext[0, :, :3, :3].cpu().numpy()                          # S,3,3 world->cam
    t = ext[0, :, :3, 3].cpu().numpy()                           # S,3
    cams = -np.einsum("sji,sj->si", R, t)                        # S,3 camera centers
    fov_h = 2.0 * np.degrees(np.arctan(H / (2.0 * intr[0, :, 1, 1].cpu().numpy())))
    fov_w = 2.0 * np.degrees(np.arctan(W / (2.0 * intr[0, :, 0, 0].cpu().numpy())))
    fov = np.stack([fov_h, fov_w], axis=1)                       # S,2 (deg)

    if point_source == "depth":
        # depth + pose unprojection: same scale as the camera trajectory, which
        # produces a continuous street.  (point head world_points use a
        # different normalized scale and ghost when mixed in.)
        dep = out["depth"][0, :, :, :, 0].float().cpu().numpy()   # S,H,W
        dcf = out.get("depth_conf")
        cf = dcf[0].float().cpu().numpy() if dcf is not None else np.ones_like(dep)
        K = intr[0, 0].cpu().numpy()
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3)
        rays = (pix @ np.linalg.inv(K).T).reshape(H, W, 3).astype(np.float32)
        p_cam = rays[None] * dep[..., None]
        wp = np.einsum("sji,shwj->shwi", R, p_cam - t[:, None, None, :]).astype(np.float32)
        return wp, cf, pred, prob, cams, R, fov

    wp = out["world_points"][0].float().cpu().numpy()            # S,H,W,3
    cf = out["world_points_conf"][0].float().cpu().numpy()       # S,H,W
    return wp, cf, pred, prob, cams, R, fov


def percentile_rank_u8(x: np.ndarray) -> np.ndarray:
    """Map every element to its percentile in [0, 255]."""
    flat = x.ravel()
    n = flat.size
    if n == 0:
        return np.zeros(0, dtype=np.uint8)
    order = np.argsort(flat, kind="stable")
    r = np.empty(n, dtype=np.float32)
    r[order] = np.linspace(0.0, 255.0, n, dtype=np.float32)
    return r.astype(np.uint8)


def extract_points(wp, cf, pred, prob, rgbs, cams, conf_q: float, seg_thresh: float,
                   drop_sky: bool, max_points: int, min_new_dist: float,
                   seed: int = 0):
    """Per-waypoint filtered clouds in display coords + reveal ranks.

    Returns ``(frames, bounds, cams_disp)`` where every frame is::

        {"points": (n,3) display, "rgb": (n,3) uint8, "cls": (n,) uint8,
         "rank": (n,) uint8, "conf": (n,) uint8, "prob": (n,) uint8}
    """
    S, H, W, _ = wp.shape
    cams_disp = display_coords(cams)
    frames = []
    total = 0
    for s in range(S):
        m = np.isfinite(wp[s]).all(-1)
        if m.any():
            thr = np.quantile(cf[s][m], float(conf_q))
            m &= cf[s] > thr
        m &= (prob[s] >= float(seg_thresh)) & (pred[s] > 0)
        if drop_sky:
            m &= pred[s] != 11

        p_disp = display_coords(wp[s][m])
        rgb = rgbs[s][m]
        cls = pred[s][m].astype(np.uint8)
        conf = percentile_rank_u8(cf[s])[m.ravel()] if m.any() else np.zeros(0, np.uint8)
        prob_u8 = np.clip(prob[s][m] * 255.0, 0, 255).astype(np.uint8)

        # near-to-far reveal rank relative to the *previous* waypoint camera
        rank = np.zeros(len(p_disp), dtype=np.float32)
        if s > 0 and len(p_disp):
            d = np.linalg.norm(p_disp - cams_disp[s - 1], axis=1)
            keep = d >= float(min_new_dist)
            idx = np.where(keep)[0]
            if len(idx):
                order = idx[np.argsort(d[idx], kind="stable")]
                rank[order] = np.linspace(0.0, 1.0, len(order), dtype=np.float32)
            rank[~keep] = 1.0
        rank_u8 = np.clip(rank * 255.0, 0, 255).astype(np.uint8)

        frames.append({"points": p_disp.astype(np.float32), "rgb": rgb.astype(np.uint8),
                       "cls": cls, "rank": rank_u8, "conf": conf, "prob": prob_u8})
        total += len(p_disp)

    # Global budget: keep a proportional random subset of every waypoint so the
    # incremental reveal stays balanced (the original script used a global
    # random subset, which could empty a whole waypoint).
    if total > int(max_points) > 0:
        frac = float(max_points) / float(total)
        rng = np.random.default_rng(seed)
        for f in frames:
            n = len(f["points"])
            if n == 0:
                continue
            k = max(1, int(round(n * frac)))
            if k < n:
                keep = np.sort(rng.choice(n, size=k, replace=False))
                for key in ("points", "rgb", "cls", "rank", "conf", "prob"):
                    f[key] = f[key][keep]

    allp = np.concatenate([f["points"] for f in frames], 0)
    lo = np.percentile(allp, 1, axis=0)
    hi = np.percentile(allp, 99, axis=0)
    pad = (hi - lo) * 0.05 + 1e-6
    bounds = (lo - pad, hi + pad)
    return frames, bounds, cams_disp


def smooth_route(cams_disp: np.ndarray, n: int = 500):
    """Arc-length parameterized PCHIP route (fallback: linear resample)."""
    try:
        from scipy.interpolate import PchipInterpolator
    except Exception:  # pragma: no cover
        PchipInterpolator = None
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(cams_disp, axis=0), axis=1))]
    if len(cams_disp) == 1 or d[-1] < 1e-9:
        return cams_disp.copy(), d
    t = np.linspace(0.0, float(d[-1]), int(n))
    if PchipInterpolator is not None:
        spl = PchipInterpolator(d, cams_disp, axis=0)
        route = spl(t)
    else:
        route = np.stack([np.interp(t, d, cams_disp[:, i]) for i in range(3)], axis=1)
    return route.astype(np.float64), t


# ---------------------------------------------------------------------------
# Binary clip cache
# ---------------------------------------------------------------------------
POINT_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ("cls", "u1"), ("wp", "u1"), ("rank", "u1"),
    ("conf", "u1"), ("prob", "u1"),
])


def pack_point_buffer(frames) -> bytes:
    """Pack per-waypoint frames into the planar buffer described in the module doc."""
    n = int(sum(len(f["points"]) for f in frames))
    buf = bytearray(20 * n)
    pos = np.concatenate([f["points"] for f in frames], 0).astype("<f4", copy=False)
    rgb = np.concatenate([f["rgb"] for f in frames], 0).astype(np.uint8, copy=False)
    cls = np.concatenate([f["cls"] for f in frames], 0).astype(np.uint8, copy=False)
    rank = np.concatenate([f["rank"] for f in frames], 0).astype(np.uint8, copy=False)
    conf = np.concatenate([f["conf"] for f in frames], 0).astype(np.uint8, copy=False)
    prob = np.concatenate([f["prob"] for f in frames], 0).astype(np.uint8, copy=False)
    wp = np.concatenate([np.full(len(f["points"]), i, dtype=np.uint8)
                         for i, f in enumerate(frames)], 0)
    buf[0:12 * n] = pos.tobytes()
    buf[12 * n:15 * n] = rgb.tobytes()
    buf[15 * n:16 * n] = cls.tobytes()
    buf[16 * n:17 * n] = wp.tobytes()
    buf[17 * n:18 * n] = rank.tobytes()
    buf[18 * n:19 * n] = conf.tobytes()
    buf[19 * n:20 * n] = prob.tobytes()
    return bytes(buf)


def load_point_buffer(path):
    """Read a packed buffer back into numpy arrays (used by the PLY exporter)."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    n = len(raw) // 20
    if len(raw) % 20 or n == 0:
        raise ValueError(f"坏的点云缓存: {path} ({len(raw)} bytes)")
    pos = np.frombuffer(raw[:12 * n].tobytes(), dtype="<f4").reshape(n, 3)
    rgb = np.frombuffer(raw[12 * n:15 * n].tobytes(), dtype=np.uint8).reshape(n, 3)
    cls = np.frombuffer(raw[15 * n:16 * n].tobytes(), dtype=np.uint8)
    return {"points": pos, "rgb": rgb, "cls": cls,
            "wp": np.frombuffer(raw[16 * n:17 * n].tobytes(), dtype=np.uint8),
            "rank": np.frombuffer(raw[17 * n:18 * n].tobytes(), dtype=np.uint8),
            "conf": np.frombuffer(raw[18 * n:19 * n].tobytes(), dtype=np.uint8),
            "prob": np.frombuffer(raw[19 * n:20 * n].tobytes(), dtype=np.uint8)}


def write_ply(path, points, colors: np.ndarray, binary: bool = True):
    """Write a binary little-endian PLY point cloud."""
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    n = len(points)
    header = (
        "ply\n"
        f"format {'binary_little_endian' if binary else 'ascii'} 1.0\n"
        "comment generated by make_demo (SegStreamVGGT)\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    if binary:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        rec = np.empty(n, dtype=dtype)
        rec["x"], rec["y"], rec["z"] = points[:, 0], points[:, 1], points[:, 2]
        rec["red"], rec["green"], rec["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
        with open(path, "wb") as fh:
            fh.write(header.encode("ascii"))
            rec.tofile(fh)
    else:
        with open(path, "w") as fh:
            fh.write(header)
            for p, c in zip(points, colors):
                fh.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n")


def write_clip(out_dir, frames, bounds, cams_disp, route, meta_extra: dict | None = None):
    """Write meta.json / points.bin / thumbs for one reconstruction clip."""
    out = Path(out_dir)
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    meta = dict(meta_extra or {})
    n = int(sum(len(f["points"]) for f in frames))
    counts, offsets, acc = [], [], 0
    for f in frames:
        offsets.append(acc)
        counts.append(int(len(f["points"])))
        acc += len(f["points"])

    with open(out / "points.bin", "wb") as fh:
        fh.write(pack_point_buffer(frames))

    # class statistics
    all_cls = np.concatenate([f["cls"] for f in frames], 0) if frames else np.zeros(0, np.uint8)
    class_counts = [int((all_cls == i).sum()) for i in range(12)]

    names = meta.get("frame_names") or [f"f_{i:03d}" for i in range(len(frames))]
    rotations = meta.get("rotations") or [None] * len(frames)
    fovs = meta.get("fovs") or [None] * len(frames)
    waypoints = []
    for i, (f, c) in enumerate(zip(frames, cams_disp)):
        waypoints.append({
            "i": i,
            "name": names[i],
            "count": counts[i],
            "offset": offsets[i],
            "cam": [float(v) for v in c],
            "R": rotations[i],
            "fov": fovs[i],
            "thumb": f"thumbs/w{i:03d}.jpg",
        })

    meta.update({
        "num_points": n,
        "num_waypoints": len(frames),
        "counts": counts,
        "offsets": offsets,
        "bounds": [[float(v) for v in bounds[0]], [float(v) for v in bounds[1]]],
        "route": [[float(v) for v in p] for p in route],
        "waypoints": waypoints,
        "class_counts": class_counts,
        "palette": PALETTE_LIST,
        "class_names": CLASS_NAMES,
        "class_labels_zh": CLASS_LABELS_ZH,
        "buffer": {"path": "points.bin", "stride": 20, "layout": "planar"},
    })
    meta.pop("frame_names", None)
    meta.pop("rotations", None)
    meta.pop("fovs", None)
    tmp = out / "meta.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    tmp.replace(out / "meta.json")
    return meta


def save_thumbs(frame_paths, out_dir, width: int = 720, quality: int = 88):
    """Save browser-friendly JPEG thumbs (one per waypoint)."""
    out = Path(out_dir) / "thumbs"
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for i, p in enumerate(frame_paths):
        p = Path(p)
        im = Image.open(p).convert("RGB")
        if im.width > width:
            im = im.resize((width, max(1, round(im.height * width / im.width))), Image.LANCZOS)
        im.save(out / f"w{i:03d}.jpg", quality=quality, optimize=True)
        names.append(p.stem)
    return names


def clip_summary(meta: dict, clip_id: str) -> dict:
    return {
        "id": clip_id,
        "label": meta.get("label", clip_id),
        "created": meta.get("created"),
        "num_waypoints": meta.get("num_waypoints", 0),
        "num_points": meta.get("num_points", 0),
        "input_hw": meta.get("input_hw"),
        "params": meta.get("params", {}),
        "source": meta.get("source", {}),
        "thumb": meta.get("waypoints", [{}])[0].get("thumb") if meta.get("waypoints") else None,
        "class_counts": meta.get("class_counts"),
        "duration_s": meta.get("duration_s"),
    }


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
