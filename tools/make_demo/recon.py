#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reconstruction worker: run the model once for one clip and write a web cache.

Called by ``server.py`` as a subprocess (one GPU-heavy task at a time)::

    python recon.py --request /path/req.json --out data/clips/<clip_id> --device cuda:0

It prints machine-readable progress lines (``@@PROGRESS@@ {...}``) which the
server forwards to the browser, plus normal human-readable logs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

import core

# The global attention mask is (S*P) x (S*P); at 100 x 1400 tokens it alone
# needs ~73 GiB.  Keep the selected-frame UX at up to 1000, but cap the actual
# single-pass inference by a conservative token budget.
MAX_INFER_TOKENS = int(os.environ.get("DEMO_MAX_INFER_TOKENS", "47000"))
HARD_MAX_INFER_FRAMES = int(os.environ.get("DEMO_MAX_INFER_FRAMES", "32"))


def infer_frame_cap(H: int, W: int):
    """Return (frame_cap, tokens_per_frame) for a single forward pass."""
    tokens = math.ceil(H / 14) * math.ceil(W / 14)
    cap = max(8, min(HARD_MAX_INFER_FRAMES, MAX_INFER_TOKENS // max(1, tokens)))
    return cap, tokens


def emit_progress(pct: float, stage: str, msg: str = ""):
    print("@@PROGRESS@@ " + json.dumps({"pct": round(float(pct), 2), "stage": stage,
                                        "msg": msg}, ensure_ascii=False), flush=True)


def log(msg: str):
    print(f"[RECON] {msg}", flush=True)


def ffprobe_video(path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
           "-show_entries", "format=duration", "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    st = (info.get("streams") or [{}])[0]
    fps = 25.0
    rate = st.get("r_frame_rate") or "25/1"
    try:
        num, den = rate.split("/")
        fps = float(num) / max(1.0, float(den))
    except Exception:
        pass
    duration = st.get("duration") or info.get("format", {}).get("duration") or 0.0
    duration = float(duration or 0.0)
    nb = st.get("nb_frames")
    try:
        nb = int(nb)
    except Exception:
        nb = int(round(duration * fps))
    return {"fps": fps, "duration": duration, "nb_frames": nb,
            "width": st.get("width"), "height": st.get("height")}


def extract_video_frames(video, indices, out_dir, max_width: int = 1400):
    """Extract requested frame indices with retries and neighbour fallback.

    Fast seeks (``-ss`` before ``-i``) are quick but can return no frame for the
    last PTS of a clip or for VFR/rounded timestamps.  We retry the same frame
    with an accurate seek and then try neighbouring indices.  A single bad frame
    is skipped instead of failing the whole reconstruction.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = ffprobe_video(video)
    fps = float(info.get("fps") or 25.0)
    duration = float(info.get("duration") or 0.0)
    nb = int(info.get("nb_frames") or 0)
    paths = []
    skipped = []
    total = len(indices)

    def run_one(dst: Path, ii: int, accurate: bool) -> bool:
        t = float(ii) / fps
        if duration > 0:
            t = min(t, max(0.0, duration - 0.001))
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-an"]
        if not accurate:
            cmd += ["-ss", f"{t:.6f}"]
        cmd += ["-i", str(video)]
        if accurate:
            cmd += ["-ss", f"{t:.6f}"]
        cmd += ["-frames:v", "1", "-vf", "scale=w='min(iw,1400)':h=-2", str(dst)]
        try:
            subprocess.run(cmd, check=True, timeout=120)
        except Exception as exc:  # noqa: BLE001
            log(f"[WARN] 抽帧失败 idx={ii} accurate={accurate}: {exc}")
            try:
                dst.unlink()
            except Exception:
                pass
            return False
        return dst.exists() and dst.stat().st_size > 100

    for j, raw_idx in enumerate(indices):
        idx = int(raw_idx)
        # exact, previous, next, two frames away, ... within the clip
        cand = []
        for delta in (0, -1, 1, -2, 2, -3, 3):
            ii = idx + delta
            if ii < 0:
                continue
            if nb > 0 and ii >= nb:
                continue
            if nb <= 0 and duration > 0 and ii / fps > duration + 1.0:
                continue
            if ii not in cand:
                cand.append(ii)
        dst = out_dir / f"f_{j:03d}.png"
        picked = None
        # Try fast then accurate for each candidate before moving further away.
        for ii in cand:
            for accurate in (False, True):
                try:
                    dst.unlink()
                except Exception:
                    pass
                if run_one(dst, ii, accurate):
                    picked = ii
                    break
            if picked is not None:
                break
        if picked is None:
            skipped.append(idx)
            log(f"[WARN] 跳过无法抽取的帧 index={idx} t={idx / fps:.3f}s")
            continue
        paths.append(dst)
        if (j + 1) % 4 == 0 or j + 1 == total:
            log(f"抽帧 {j + 1}/{total} (t={picked / fps:.2f}s)")

    if len(paths) < 2:
        raise RuntimeError(f"视频抽帧失败：仅得到 {len(paths)}/{total} 帧，"
                           f"无法重建（indices={list(indices)[:8]}...）")
    if skipped:
        log(f"[WARN] 共跳过 {len(skipped)} 帧: {skipped[:12]}")
    return paths, info


def resolve_frames(req: dict, max_frames: int = 0):
    """Return (frame_paths, source_meta, video_info)."""
    src = req["source"]
    kind = src.get("kind")
    video_info = None
    if kind == "frames":
        paths = [Path(p) for p in src["paths"]]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"以下输入帧不存在: {missing[:5]}")
        requested = len(paths)
        if max_frames and len(paths) > max_frames:
            paths = core.select_evenly(paths, max_frames)
        return paths, {"kind": "frames", "paths": [str(p) for p in paths],
                       "root": src.get("root"), "requested_frames": requested}, video_info
    if kind == "images":
        paths = [Path(p) for p in src["paths"]]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"以下上传图片不存在: {missing[:5]}")
        requested = len(paths)
        if max_frames and len(paths) > max_frames:
            paths = core.select_evenly(paths, max_frames)
        return paths, {"kind": "images", "count": len(paths),
                       "paths": [str(p) for p in paths],
                       "upload_id": src.get("upload_id"),
                       "requested_frames": requested}, video_info
    if kind == "video":
        video = Path(src["video"])
        indices = [int(i) for i in src["indices"]]
        requested = len(indices)
        if max_frames and len(indices) > max_frames:
            indices = core.select_evenly(indices, max_frames)
        work = Path(req["work_dir"])
        paths, info = extract_video_frames(video, indices, work / "frames")
        video_info = info
        return paths, {"kind": "video", "video": str(video),
                       "indices": indices, "fps": info["fps"],
                       "duration": info["duration"],
                       "upload_id": src.get("upload_id"),
                       "requested_frames": requested}, video_info
    raise ValueError(f"未知的 source.kind: {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    t0 = time.time()
    req = json.loads(Path(args.request).read_text(encoding="utf-8"))
    params = req.get("params", {})
    H, W = core.parse_hw(params.get("img_size", "392x700"))
    if H % 14 or W % 14:
        raise ValueError(f"img_size 必须是 14 的倍数, 收到 {H}x{W}")
    point_source = params.get("point_source", "depth")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    emit_progress(1, "start", "准备输入帧")
    infer_cap, tokens_per_frame = infer_frame_cap(H, W)
    requested_max = int(params.get("max_frames") or 0)
    effective_max = min(requested_max, infer_cap) if requested_max else infer_cap
    frame_paths, source_meta, video_info = resolve_frames(req, max_frames=effective_max)
    if not frame_paths:
        raise RuntimeError("没有输入帧")
    requested = int(source_meta.get("requested_frames") or len(frame_paths))
    effective_max = min(effective_max, requested)
    if requested > len(frame_paths):
        log(f"显存保护：选择 {requested} 帧 -> 均匀降采样到 {len(frame_paths)} 帧 "
            f"（{H}x{W}, {tokens_per_frame} tokens/帧, 单次预算 {MAX_INFER_TOKENS}）")
    else:
        log(f"单次推理 {len(frame_paths)} 帧（{H}x{W}, {tokens_per_frame} tokens/帧, cap {infer_cap}）")
    log(f"输入 {len(frame_paths)} 帧, 目标分辨率 {H}x{W}")
    emit_progress(4, "frames", f"已准备 {len(frame_paths)} 帧" +
                  (f"（{requested} 帧均匀降采样）" if requested > len(frame_paths) else ""))

    ckpt = req.get("ckpt") or core.DEFAULT_CKPT
    backbone = req.get("backbone") or core.DEFAULT_BACKBONE
    out_ratio = int(params.get("out_ratio", 2))

    emit_progress(8, "model", "加载 StreamVGGT + SegStreamVGGT 权重（约 20-40s）")
    model = core.build_model(ckpt, backbone, out_ratio, args.device, point_source)
    emit_progress(40, "model", "模型就绪")
    log(f"模型加载完成 {time.time() - t0:.1f}s")

    images, rgbs, names = core.load_frame_files(frame_paths, H, W)
    images = images.to(args.device)
    emit_progress(46, "infer", f"前向推理 {len(frame_paths)} 帧（冻结几何 + 语义头）")
    wp, cf, pred, prob, cams, R, fov = core.infer(model, images, H, W, point_source)
    log(f"推理完成 {time.time() - t0:.1f}s, world_points={wp.shape}")
    emit_progress(74, "infer", "推理完成，正在过滤/反投影点位")

    frames, bounds, cams_disp = core.extract_points(
        wp, cf, pred, prob, rgbs, cams,
        conf_q=float(params.get("conf_quantile", 0.10)),
        seg_thresh=float(params.get("seg_thresh", 0.0)),
        drop_sky=bool(params.get("drop_sky", True)),
        max_points=int(params.get("max_points", 1500000)),
        min_new_dist=float(params.get("min_new_dist", 0.15)),
    )
    counts = [len(f["points"]) for f in frames]
    log(f"每航点点数: {counts} (总 {sum(counts):,})")
    emit_progress(84, "points", f"保留 {sum(counts):,} 个点")

    emit_progress(88, "route", "平滑航线 / 生成缩略图")
    route, _ = core.smooth_route(cams_disp, n=500)
    core.save_thumbs(frame_paths, out, width=720)

    created = datetime.now().isoformat(timespec="seconds")
    label = req.get("label") or Path(frame_paths[0]).parent.name or "clip"
    meta_extra = {
        "id": req.get("clip_id"),
        "label": label,
        "created": created,
        "source": source_meta,
        "params": {
            "img_size": f"{H}x{W}", "max_frames": effective_max,
            "conf_quantile": float(params.get("conf_quantile", 0.10)),
            "seg_thresh": float(params.get("seg_thresh", 0.0)),
            "max_points": int(params.get("max_points", 1500000)),
            "point_source": point_source,
            "drop_sky": bool(params.get("drop_sky", True)),
            "min_new_dist": float(params.get("min_new_dist", 0.15)),
            "out_ratio": out_ratio,
        },
        "model": {"ckpt": str(ckpt), "model_label": req.get("model_label", "v5"),
                  "backbone": str(backbone)},
        "input_hw": [H, W],
        "frame_names": names,
        "rotations": [[float(v) for v in r.ravel()] for r in R],
        "fovs": [[float(a), float(b)] for a, b in fov],
        "camera_centers": [[float(v) for v in c] for c in cams],
        "duration_s": round(time.time() - t0, 2),
        "video": video_info,
    }
    emit_progress(93, "write", "写入点云缓存")
    core.write_clip(out, frames, bounds, cams_disp, route, meta_extra)

    emit_progress(100, "done", f"完成：{len(frames)} 航点 / {sum(counts):,} 点")
    print("@@RESULT@@ " + json.dumps({"out": str(out), "points": int(sum(counts)),
                                      "waypoints": len(frames)}, ensure_ascii=False), flush=True)
    log(f"完成，用时 {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        emit_progress(100, "error", f"{type(exc).__name__}: {exc}")
        sys.exit(1)
