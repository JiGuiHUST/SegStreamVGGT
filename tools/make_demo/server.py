#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interactive web backend for the SegStreamVGGT UAV city demo.

Features
--------
* serve reconstructed clips (meta.json + points.bin + thumbnails)
* run new reconstructions from a frame folder / uploaded images / uploaded video
  in a background subprocess (one GPU job at a time)
* browse server-side frame folders, extract video candidate frames
* export PLY on demand

Run::

    cd /data/luhangze/aic/make_demo
    python server.py --host 0.0.0.0 --port 8787   # --gpu auto (default)

Then open http://<host>:8787 .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List
from urllib.parse import quote

import numpy as np
from PIL import Image

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DEMO_DATA_DIR", APP_DIR / "data")).resolve()
CLIPS_DIR = DATA_DIR / "clips"
UPLOADS_DIR = DATA_DIR / "uploads"
WORK_DIR = DATA_DIR / "work"
JOBS_DIR = WORK_DIR / "jobs"
THUMB_CACHE = WORK_DIR / "thumbs"
for d in (CLIPS_DIR, UPLOADS_DIR, JOBS_DIR, THUMB_CACHE):
    d.mkdir(parents=True, exist_ok=True)

import core  # noqa: E402  (light import: numpy/PIL only)

DEFAULT_ROOTS = ["/data/luhangze", "/data1", "/tmp", str(DATA_DIR)]
IMAGE_EXTS = core.IMAGE_EXTS
CKPT = core.DEFAULT_CKPT
BACKBONE = core.DEFAULT_BACKBONE
GPU = os.environ.get("DEMO_GPU", "auto")
MAX_FRAMES = 1000
MAX_INFER_FRAMES = int(os.environ.get("DEMO_MAX_INFER_FRAMES", "32"))
MAX_INFER_TOKENS = int(os.environ.get("DEMO_MAX_INFER_TOKENS", "47000"))
GPU_MIN_FREE_MB = int(os.environ.get("DEMO_GPU_MIN_FREE_MB", "16000"))
GPU_UTIL_PENALTY_MB = int(os.environ.get("DEMO_GPU_UTIL_PENALTY_MB", "300"))
PYTHON = os.environ.get("DEMO_PYTHON", "/data/luhangze/miniconda3/envs/segvggt/bin/python")

PRESET_DIRS = [
    ("city2 · 城市街区", "/data/luhangze/aic/demo/city2_seq/frames_route", "16 帧 / 5s 航线，当前样例"),
    ("uavid seq16 · 街道", "/data/luhangze/aic/SegStreamVGGT/demo/uavid_seq16/frames_route", "原始 UAVid 航迹 demo"),
    ("train_seq2 · 街区", "/data/luhangze/aic/SegStreamVGGT/demo/clips/train_seq2/frames", "训练序列 2"),
    ("train_seq3 · 街区", "/data/luhangze/aic/SegStreamVGGT/demo/clips/train_seq3/frames", "训练序列 3"),
    ("test_seq21 · 街区", "/data/luhangze/aic/SegStreamVGGT/demo/clips/test_seq21/frames", "测试序列 21"),
    ("test_seq22 · 街区", "/data/luhangze/aic/SegStreamVGGT/demo/clips/test_seq22/frames", "测试序列 22"),
    ("372124 · 城市视频", "/data/luhangze/aic/SegStreamVGGT/demo/372124/frames", "24 帧"),
    ("city2 全帧", "/data/luhangze/aic/demo/city2_frames_all", "148 帧，可自选航点"),
]

ALLOWED_ROOTS = [Path(p).resolve() for p in
                 os.environ.get("DEMO_ALLOWED_ROOTS", ":".join(DEFAULT_ROOTS)).split(":") if p]

app = FastAPI(title="AIC UAV Interactive Demo", docs_url="/api/docs", redoc_url=None)


app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
app.mount("/clips", StaticFiles(directory=CLIPS_DIR), name="clips")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def safe_path(raw: str, must_exist: bool = True) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (APP_DIR / p).resolve()
    else:
        p = p.resolve()
    if not any(_is_under(p, r) for r in ALLOWED_ROOTS):
        raise HTTPException(403, f"路径不在允许的根目录内: {p}")
    if must_exist and not p.exists():
        raise HTTPException(404, f"路径不存在: {p}")
    return p


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def list_images_sorted(p: Path) -> List[Path]:
    return [q for q in sorted(p.iterdir()) if is_image(q)]


def ffprobe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
           "-show_entries", "format=duration", "-of", "json", str(path)]
    try:
        info = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    except Exception:
        return {"fps": 25.0, "duration": 0.0, "nb_frames": 0, "width": None, "height": None}
    st = (info.get("streams") or [{}])[0]
    fps = 25.0
    try:
        num, den = (st.get("r_frame_rate") or "25/1").split("/")
        fps = float(num) / max(1.0, float(den))
    except Exception:
        pass
    duration = float(st.get("duration") or info.get("format", {}).get("duration") or 0.0)
    try:
        nb = int(st.get("nb_frames"))
    except Exception:
        nb = int(round(duration * fps))
    return {"fps": round(fps, 4), "duration": round(duration, 3), "nb_frames": nb,
            "width": st.get("width"), "height": st.get("height")}


def make_thumb(src: Path, dst: Path, width: int = 360):
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    if im.width > width:
        im = im.resize((width, max(1, round(im.height * width / im.width))), Image.LANCZOS)
    tmp = dst.with_suffix(".tmp.jpg")
    im.save(tmp, quality=86, optimize=True)
    tmp.replace(dst)


def cached_thumb(path: Path, width: int) -> Path:
    st = path.stat()
    key = hashlib.sha1(f"{path}|{st.st_mtime_ns}|{st.st_size}|{width}".encode()).hexdigest()[:24]
    dst = THUMB_CACHE / f"{key}.jpg"
    make_thumb(path, dst, width)
    return dst


def clip_dir(clip_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", clip_id or ""):
        raise HTTPException(400, "非法 clip id")
    p = (CLIPS_DIR / clip_id).resolve()
    if not _is_under(p, CLIPS_DIR) or not p.is_dir():
        raise HTTPException(404, "片段不存在")
    return p


def read_meta(clip_id: str) -> dict:
    p = clip_dir(clip_id) / "meta.json"
    if not p.exists():
        raise HTTPException(404, "meta.json 不存在")
    return json.loads(p.read_text(encoding="utf-8"))


def clip_summaries() -> list:
    out = []
    for d in CLIPS_DIR.iterdir():
        if not d.is_dir():
            continue
        mp = d / "meta.json"
        if not mp.exists():
            continue
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        item = core.clip_summary(meta, d.name)
        src = item.get("source") or {}
        item["source_kind"] = src.get("kind")
        item["rebuildable"] = bool(src.get("kind") in ("frames", "video")
                                   or src.get("paths") or src.get("upload_id"))
        out.append(item)
    out.sort(key=lambda x: x.get("created") or "", reverse=True)
    return out


def slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", (text or "").lower()).strip("-")
    if not re.search(r"[a-z]", s):
        s = ("clip-" + s).strip("-") if s else "clip"
    return s[:40] or "clip"


def new_clip_id(label: str) -> str:
    base = slugify(label)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    cid = f"{base}-{stamp}"
    i = 2
    while (CLIPS_DIR / cid).exists():
        cid = f"{base}-{stamp}-{i}"
        i += 1
    return cid


def list_gpus() -> list:
    """Best-effort nvidia-smi snapshot; [] when the tool is unavailable."""
    cmd = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,"
                          "utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True).stdout
    except Exception:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            idx = int(parts[0])
            total = int(float(parts[3]))
            used = int(float(parts[4]))
            free = int(float(parts[5]))
            util = 0 if parts[6] in ("N/A", "[N/A]") else int(float(parts[6]))
            temp = 0 if parts[7] in ("N/A", "[N/A]") else int(float(parts[7]))
        except Exception:
            continue
        gpus.append({"index": idx, "uuid": parts[1], "name": parts[2],
                     "total_mb": total, "used_mb": used, "free_mb": free,
                     "util": util, "temp": temp})
    return gpus


def pick_gpu() -> dict:
    """Pick the least-loaded GPU at job start.

    ``DEMO_GPU``/``--gpu`` may still pin a physical index for debugging; the
    default is ``auto``.  With auto, free memory is preferred, with a penalty
    for GPUs already under high utilization (so a 100%-busy 40 GB card does not
    beat an idle 35 GB card).
    """
    mode = str(GPU or "auto").strip()
    if mode.lower() not in ("auto", "none", ""):
        return {"index": mode, "mode": "manual", "reason": f"manual GPU {mode}"}
    gpus = list_gpus()
    if not gpus:
        return {"index": "0", "mode": "auto", "reason": "nvidia-smi unavailable; fallback GPU 0"}
    eligible = [g for g in gpus if g["free_mb"] >= GPU_MIN_FREE_MB] or gpus
    best = max(eligible, key=lambda g: (g["free_mb"] - g["util"] * GPU_UTIL_PENALTY_MB,
                                        g["free_mb"], -g["index"]))
    reason = (f"auto: free={best['free_mb']}MB util={best['util']}% "
              f"temp={best['temp']}C among {len(gpus)} GPUs")
    return {"index": str(best["index"]), "mode": "auto", "reason": reason,
            "uuid": best["uuid"], "name": best["name"],
            "free_mb": best["free_mb"], "util": best["util"], "temp": best["temp"]}


def log_job(job: dict, line: str):
    line = line.rstrip()
    if not line:
        return
    job["logs"].append(line)
    if len(job["logs"]) > 400:
        del job["logs"][:-400]


def job_view(job: dict) -> dict:
    return {
        "id": job["id"], "clip_id": job["clip_id"], "label": job["label"],
        "status": job["status"], "pct": job["pct"], "stage": job["stage"], "msg": job["msg"],
        "created": job["created"], "started": job.get("started"), "finished": job.get("finished"),
        "error": job.get("error"), "logs": job["logs"][-60:],
        "queue_pos": job.get("queue_pos", 0), "result": job.get("result"),
        "gpu": job.get("gpu"), "gpu_info": job.get("gpu_info"),
    }


# ---------------------------------------------------------------------------
# job runner
# ---------------------------------------------------------------------------
def _run_job(job: dict):
    with JOB_LOCK:
        job.update(status="running", started=time.time(), pct=1.0, stage="starting",
                   msg="启动重建进程")
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_CACHE.mkdir(parents=True, exist_ok=True)
    req_path = JOBS_DIR / f"{job['id']}.req.json"
    req_path.write_text(json.dumps(job["request"], ensure_ascii=False, indent=2), encoding="utf-8")
    out_dir = CLIPS_DIR / job["clip_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_choice = pick_gpu()
    gpu_index = gpu_choice["index"]
    job["gpu"] = gpu_index
    job["gpu_info"] = gpu_choice
    log_job(job, f"$ GPU {gpu_index} [{gpu_choice['reason']}]")
    cmd = [PYTHON, str(APP_DIR / "recon.py"), "--request", str(req_path),
           "--out", str(out_dir), "--device", "cuda:0"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["PYTHONUNBUFFERED"] = "1"
    log_job(job, f"$ CUDA_VISIBLE_DEVICES={gpu_index} {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(APP_DIR), env=env, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    job["pid"] = proc.pid
    canceled = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if job.get("cancel_requested"):
                canceled = True
                proc.terminate()
                break
            if line.startswith("@@PROGRESS@@ "):
                try:
                    ev = json.loads(line[len("@@PROGRESS@@ "):])
                except Exception:
                    continue
                with JOB_LOCK:
                    job["pct"] = float(ev.get("pct", job["pct"]))
                    job["stage"] = ev.get("stage", job["stage"])
                    job["msg"] = ev.get("msg", job["msg"])
                if ev.get("msg"):
                    log_job(job, f"[{ev.get('stage')}] {ev.get('msg')}")
            elif line.startswith("@@RESULT@@ "):
                try:
                    job["result"] = json.loads(line[len("@@RESULT@@ "):])
                except Exception:
                    pass
            else:
                log_job(job, line)
        rc = proc.wait(timeout=60)
    except Exception as exc:  # noqa: BLE001
        log_job(job, f"进程异常: {exc}")
        try:
            proc.kill()
        except Exception:
            pass
        rc = -1

    meta_file = out_dir / "meta.json"
    if canceled or job.get("cancel_requested"):
        with JOB_LOCK:
            job.update(status="canceled", pct=0, stage="canceled", msg="已取消",
                       finished=time.time())
        shutil.rmtree(out_dir, ignore_errors=True)
        return
    if rc == 0 and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        with JOB_LOCK:
            job.update(status="done", pct=100.0, stage="done", msg="重建完成",
                       finished=time.time(), result={
                           "clip": core.clip_summary(meta, job["clip_id"]),
                           "points": meta.get("num_points"),
                           "waypoints": meta.get("num_waypoints")})
        log_job(job, f"完成 -> {job['clip_id']}")
    else:
        with JOB_LOCK:
            job.update(status="error", pct=100.0, stage="error", finished=time.time(),
                       msg="重建失败，请查看日志", error=f"recon.py 退出码 {rc}")
        log_job(job, f"失败，退出码 {rc}")
        shutil.rmtree(out_dir, ignore_errors=True)


def _worker_loop():
    while True:
        jid = JOB_QUEUE.get()
        try:
            job = JOBS.get(jid)
            if not job:
                continue
            if job.get("cancel_requested") or job["status"] == "canceled":
                with JOB_LOCK:
                    job.update(status="canceled", msg="已取消")
                continue
            _run_job(job)
        except Exception as exc:  # noqa: BLE001
            with JOB_LOCK:
                if jid in JOBS:
                    JOBS[jid].update(status="error", error=str(exc), msg=str(exc))
        finally:
            JOB_QUEUE.task_done()


def create_job(payload: dict) -> dict:
    with JOB_LOCK:
        active = [j for j in JOBS.values() if j["status"] in ("queued", "running")]
        if len(active) >= 8:
            raise HTTPException(429, "排队任务过多，请等待当前重建完成")
        jid = uuid.uuid4().hex[:12]
        clip_id = new_clip_id(payload.get("label") or "clip")
        request = dict(payload)
        request["clip_id"] = clip_id
        base_work = Path(request.get("work_dir") or (WORK_DIR / "extract"))
        request["work_dir"] = str(base_work / jid)
        job = {
            "id": jid, "clip_id": clip_id, "label": payload.get("label") or clip_id,
            "status": "queued", "pct": 0.0, "stage": "queued", "msg": "排队中",
            "created": time.time(), "logs": [], "request": request,
            "queue_pos": len(active) + 1,
        }
        JOBS[jid] = job
    JOB_QUEUE.put(jid)
    return job


# ---------------------------------------------------------------------------
# reconstruction request helpers
# ---------------------------------------------------------------------------
def clean_recon_params(params: dict) -> dict:
    """Validate/normalise the subset of recon parameters exposed by the UI."""
    params = params or {}

    def fnum(key, default, lo, hi):
        try:
            v = float(params.get(key, default))
        except Exception:
            v = float(default)
        return max(lo, min(hi, v))

    img_size = str(params.get("img_size") or "392x700").lower()
    try:
        H, W = core.parse_hw(img_size)
    except Exception:
        raise HTTPException(400, "img_size 格式应为 HxW，例如 392x700")
    if H % 14 or W % 14:
        raise HTTPException(400, "img_size 的高和宽必须是 14 的倍数")
    if not (112 <= H <= 1036 and 112 <= W <= 1036):
        raise HTTPException(400, "img_size 超出范围")
    point_source = params.get("point_source", "depth")
    if point_source not in ("depth", "point"):
        raise HTTPException(400, "point_source 必须是 depth 或 point")
    return {
        "img_size": f"{H}x{W}",
        "max_frames": int(max(0, min(MAX_FRAMES, int(params.get("max_frames") or 0)))),
        "conf_quantile": fnum("conf_quantile", 0.10, 0.0, 0.9),
        "seg_thresh": fnum("seg_thresh", 0.0, 0.0, 0.95),
        "max_points": int(max(50000, min(4000000, int(params.get("max_points") or 1500000)))),
        "min_new_dist": fnum("min_new_dist", 0.15, 0.0, 5.0),
        "point_source": point_source,
        "drop_sky": bool(params.get("drop_sky", True)),
        "out_ratio": 2,
    }


def source_for_rebuild(meta: dict) -> dict:
    """Rebuild the normalised source object from a clip's stored meta."""
    src = meta.get("source") or {}
    kind = src.get("kind")
    if kind == "frames":
        raw = [str(x) for x in (src.get("paths") or [])]
        if not raw:
            raise HTTPException(409, "该场景未保存原始帧路径，无法重建")
        if len(raw) > MAX_FRAMES:
            raw = raw[:MAX_FRAMES]
        paths = []
        for rp in raw:
            try:
                p = safe_path(rp)
            except HTTPException:
                raise HTTPException(409, f"原始帧已不存在或不可访问: {rp}")
            if not is_image(p):
                raise HTTPException(409, f"原始输入不是图片: {p}")
            paths.append(str(p))
        if len(paths) < 2:
            raise HTTPException(409, "原始帧少于 2 张，无法重建")
        return {"kind": "frames", "paths": paths,
                "root": str(src.get("root") or Path(paths[0]).parent)}
    if kind == "video":
        raw = str(src.get("video") or "")
        if not raw:
            raise HTTPException(409, "该场景未保存原始视频路径，无法重建")
        try:
            video = safe_path(raw)
        except HTTPException:
            raise HTTPException(409, f"原始视频已不存在或不可访问: {raw}")
        if not video.is_file():
            raise HTTPException(409, f"原始视频已不存在: {video}")
        indices = sorted(set(int(i) for i in (src.get("indices") or [])))
        if len(indices) < 2:
            raise HTTPException(409, "原始视频选择少于 2 帧，无法重建")
        if len(indices) > MAX_FRAMES:
            raise HTTPException(409, f"原始视频选择超过 {MAX_FRAMES} 帧")
        return {"kind": "video", "video": str(video), "indices": indices,
                "upload_id": src.get("upload_id")}
    if kind == "images":
        raw = [str(x) for x in (src.get("paths") or [])]
        if not raw and src.get("upload_id"):
            d = (UPLOADS_DIR / str(src.get("upload_id")) / "img").resolve()
            if _is_under(d, UPLOADS_DIR) and d.is_dir():
                raw = [str(p) for p in list_images_sorted(d)]
        if not raw:
            raise HTTPException(409, "该场景创建时未保存原图路径，无法重建；请重新上传图片")
        if len(raw) > MAX_FRAMES:
            raw = raw[:MAX_FRAMES]
        paths = []
        for rp in raw:
            try:
                p = safe_path(rp)
            except HTTPException:
                raise HTTPException(409, f"原始图片已不存在或不可访问: {rp}")
            if not is_image(p):
                raise HTTPException(409, f"原始输入不是图片: {p}")
            paths.append(str(p))
        if len(paths) < 2:
            raise HTTPException(409, "原始图片少于 2 张，无法重建")
        return {"kind": "images", "paths": paths, "count": len(paths),
                "upload_id": src.get("upload_id")}
    raise HTTPException(409, "该场景没有可用的原始输入信息，无法重建")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    # Never let a browser pair an old HTML shell with a newer app.js.
    return FileResponse(APP_DIR / "static" / "index.html",
                        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "clip_count": len(clip_summaries()),
        "ckpt": CKPT, "ckpt_exists": Path(CKPT).exists(),
        "backbone": BACKBONE, "backbone_exists": Path(BACKBONE).exists(),
        "gpu": GPU, "gpu_mode": GPU, "gpus": list_gpus(), "python": PYTHON,
        "allowed_roots": [str(r) for r in ALLOWED_ROOTS],
        "jobs": {k: v["status"] for k, v in JOBS.items()},
    }


@app.get("/api/config")
def config():
    presets = []
    for label, path, desc in PRESET_DIRS:
        p = Path(path)
        if p.is_dir():
            try:
                n = len(list_images_sorted(p))
            except Exception:
                n = 0
            if n:
                presets.append({"label": label, "path": str(p), "count": n, "desc": desc})
    return {
        "app": "AIC UAV Interactive Reconstruction",
        "class_names": core.CLASS_NAMES,
        "class_labels_zh": core.CLASS_LABELS_ZH,
        "palette": core.PALETTE_LIST,
        "default_params": {
            "img_size": "392x700", "max_points": 1500000, "conf_quantile": 0.10,
            "seg_thresh": 0.0, "min_new_dist": 0.15, "drop_sky": True,
            "point_source": "depth",
        },
        "presets": presets,
        "allowed_roots": [str(r) for r in ALLOWED_ROOTS],
        "gpu": GPU,
        "gpu_mode": GPU,
        "limits": {"max_frames": MAX_FRAMES, "video_preview_candidates": 20,
                   "max_infer_frames": MAX_INFER_FRAMES, "max_infer_tokens": MAX_INFER_TOKENS},
        "model": {"ckpt": CKPT, "label": "v5"},
    }


@app.get("/api/clips")
def api_clips():
    return {"clips": clip_summaries()}


@app.get("/api/clips/{clip_id}/meta")
def api_clip_meta(clip_id: str):
    return read_meta(clip_id)


@app.delete("/api/clips/{clip_id}")
def api_delete_clip(clip_id: str):
    d = clip_dir(clip_id)
    with JOB_LOCK:
        running = [j for j in JOBS.values()
                   if j["clip_id"] == clip_id and j["status"] in ("queued", "running")]
        if running:
            raise HTTPException(409, "该片段正在重建中")
    shutil.rmtree(d)
    return {"ok": True}


@app.post("/api/clips/{clip_id}/rebuild")
def api_rebuild_clip(clip_id: str, payload: dict = Body(default={}),
                     dry_run: bool = Query(False)):
    """Create a new reconstruction from a clip's stored source + params.

    The old clip is kept; the new job gets a new clip id.  ``dry_run=1`` only
    validates that the original source is still reachable.
    """
    clip_dir(clip_id)
    meta = read_meta(clip_id)
    source = source_for_rebuild(meta)
    params = clean_recon_params(meta.get("params") or {})
    base_label = str(meta.get("label") or clip_id)
    label = str((payload or {}).get("label") or (base_label + " · 重建")).strip()
    label = label[:80] or (base_label + " · 重建")
    if source["kind"] == "frames":
        n_frames = len(source["paths"])
    elif source["kind"] == "video":
        n_frames = len(source["indices"])
    else:
        n_frames = int(source.get("count") or len(source.get("paths") or []))
    if dry_run:
        return {"ok": True, "clip_id": clip_id, "source_kind": source["kind"],
                "frames": n_frames, "label": label, "params": params}
    model = meta.get("model") or {}
    ckpt = str(model.get("ckpt") or CKPT)
    backbone = str(model.get("backbone") or BACKBONE)
    if not Path(ckpt).exists():
        ckpt = CKPT
    if not Path(backbone).exists():
        backbone = BACKBONE
    request = {
        "label": label,
        "source": source,
        "params": params,
        "ckpt": ckpt,
        "backbone": backbone,
        "model_label": model.get("model_label", "v5"),
        "work_dir": str(WORK_DIR / "extract"),
    }
    job = create_job(request)
    return {"job_id": job["id"], "clip_id": job["clip_id"], "status": job["status"],
            "label": label, "source_kind": source["kind"], "frames": n_frames}


@app.get("/api/clips/{clip_id}/ply")
def api_clip_ply(clip_id: str, mode: str = Query("seg", pattern="^(seg|rgb|cls)$")):
    d = clip_dir(clip_id)
    meta = read_meta(clip_id)
    mode = "seg" if mode == "cls" else mode
    dst = d / f"export_{mode}.ply"
    if not dst.exists():
        data = core.load_point_buffer(d / "points.bin")
        if mode == "rgb":
            colors = data["rgb"]
        else:
            pal = np.array(meta.get("palette", core.PALETTE_LIST), dtype=np.uint8)
            colors = pal[data["cls"]]
        core.write_ply(dst, data["points"], colors)
    return FileResponse(dst, media_type="application/octet-stream",
                        filename=f"{clip_id}_{mode}.ply")


# ---- server-side frame browsing --------------------------------------------
@app.get("/api/browse")
def api_browse(path: str = Query(...)):
    p = safe_path(path)
    if p.is_file():
        p = p.parent
    if not p.is_dir():
        raise HTTPException(400, "不是目录")
    try:
        entries = sorted(p.iterdir(), key=lambda q: q.name.lower())
    except PermissionError:
        raise HTTPException(403, "目录不可读")
    dirs, images = [], []
    for q in entries:
        if q.name.startswith("."):
            continue
        if q.is_dir():
            dirs.append({"name": q.name, "path": str(q)})
        elif is_image(q):
            images.append(q)
        if len(dirs) > 300 or len(images) > 500:
            break
    items = []
    for q in images[:300]:
        items.append({
            "name": q.name, "path": str(q), "size": q.stat().st_size,
            "url": f"/api/frames/full?path={quote(str(q), safe='')}",
            "thumb": f"/api/frames/thumb?path={quote(str(q), safe='')}&w=260",
        })
    return {"path": str(p), "parent": str(p.parent), "dirs": dirs, "images": items,
            "image_total": len(images), "truncated": len(images) > 300}


@app.get("/api/frames/thumb")
def api_frame_thumb(path: str = Query(...), w: int = Query(320, ge=64, le=1280)):
    p = safe_path(path)
    if not is_image(p):
        raise HTTPException(400, "不是图片")
    return FileResponse(cached_thumb(p, w), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/frames/full")
def api_frame_full(path: str = Query(...), w: int = Query(1400, ge=128, le=4096)):
    p = safe_path(path)
    if not is_image(p):
        raise HTTPException(400, "不是图片")
    return FileResponse(cached_thumb(p, w), media_type="image/jpeg")


# ---- uploads ---------------------------------------------------------------
@app.post("/api/upload/video")
async def api_upload_video(file: UploadFile = File(...),
                           candidates: int = Query(16, ge=4, le=40)):
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}:
        raise HTTPException(400, f"不支持的视频格式: {suffix}")
    uid = uuid.uuid4().hex[:12]
    d = UPLOADS_DIR / uid
    d.mkdir(parents=True, exist_ok=True)
    video = d / f"video{suffix}"
    with open(video, "wb") as fh:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    info = ffprobe(video)
    nb = max(1, int(info.get("nb_frames") or 0))
    if info.get("duration") and nb <= 1:
        nb = max(1, int(round(info["duration"] * info["fps"])))
    cand_dir = d / "cand"
    cand_dir.mkdir(exist_ok=True)
    n = min(candidates, max(1, nb))
    idxs = sorted(set(int(round(i * (nb - 1) / max(1, n - 1))) for i in range(n))) if n > 1 else [0]
    cands = []
    used = set()
    fps = max(1e-6, float(info.get("fps") or 25.0))
    for j, idx in enumerate(idxs):
        picked = None
        dst = cand_dir / f"c_{j:03d}.jpg"
        # Exact-frame seeks can fail at the tail of a VFR clip.  Try nearby
        # frames so a 20-candidate request really returns 20 candidates.
        for delta in (0, -1, 1, -2, 2, -3, 3):
            ii = idx + delta
            if ii < 0 or ii >= nb or ii in used:
                continue
            t = ii / fps
            if dst.exists():
                try:
                    dst.unlink()
                except Exception:
                    pass
            cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t:.4f}",
                   "-i", str(video), "-frames:v", "1",
                   "-vf", "scale=w='min(iw,360)':h=-2", str(dst)]
            try:
                subprocess.run(cmd, check=True, timeout=60)
            except Exception:
                continue
            if dst.exists() and dst.stat().st_size > 100:
                picked = ii
                break
        if picked is None:
            continue
        used.add(picked)
        cands.append({"idx": picked, "t": round(picked / fps, 3),
                      "url": f"/uploads/{uid}/cand/c_{j:03d}.jpg"})
    return {"upload_id": uid, "filename": file.filename, "info": info,
            "video_url": f"/uploads/{uid}/{video.name}",
            "candidates": cands}


@app.post("/api/upload/images")
async def api_upload_images(files: List[UploadFile] = File(...)):
    uid = uuid.uuid4().hex[:12]
    d = UPLOADS_DIR / uid
    img_dir = d / "img"
    thumb_dir = d / "thumb"
    img_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for i, f in enumerate(files):
        suffix = Path(f.filename or f"img_{i}.png").suffix.lower() or ".png"
        if suffix not in IMAGE_EXTS:
            continue
        name = f"img_{i:03d}{suffix}"
        dst = img_dir / name
        with open(dst, "wb") as fh:
            while True:
                chunk = await f.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        try:
            make_thumb(dst, thumb_dir / f"{name}.jpg", 360)
        except Exception:
            continue
        items.append({"name": f.filename, "path": str(dst), "index": i,
                      "url": f"/uploads/{uid}/img/{name}",
                      "thumb": f"/uploads/{uid}/thumb/{name}.jpg"})
    if not items:
        raise HTTPException(400, "没有可用的图片")
    return {"upload_id": uid, "items": items}


# ---- reconstruction jobs ---------------------------------------------------
@app.post("/api/recon")
def api_recon(payload: dict = Body(...)):
    src = payload.get("source") or {}
    kind = src.get("kind")
    params = payload.get("params") or {}
    label = (payload.get("label") or "").strip() or "新片段"

    # ---- normalise the frame source
    if kind == "frames":
        raw_paths = [str(x) for x in (src.get("paths") or [])][:MAX_FRAMES]
        if len(raw_paths) < 2:
            raise HTTPException(400, "至少选择 2 个航点帧")
        paths = []
        for rp in raw_paths:
            p = safe_path(rp)
            if not is_image(p):
                raise HTTPException(400, f"不是图片: {p}")
            paths.append(str(p))
        source = {"kind": "frames", "paths": paths,
                  "root": str(Path(paths[0]).parent)}
    elif kind == "images":
        uid = src.get("upload_id")
        if not uid or not re.fullmatch(r"[a-f0-9]{8,32}", uid):
            raise HTTPException(400, "upload_id 非法")
        d = (UPLOADS_DIR / uid / "img").resolve()
        if not _is_under(d, UPLOADS_DIR) or not d.is_dir():
            raise HTTPException(404, "上传的图片不存在")
        selected = src.get("paths")
        if selected:
            paths = []
            for rp in selected[:MAX_FRAMES]:
                q = safe_path(rp)
                if not is_image(q) or not _is_under(q, d):
                    raise HTTPException(400, f"非法上传图片: {q}")
                paths.append(str(q))
        else:
            paths = [str(p) for p in list_images_sorted(d)][:MAX_FRAMES]
        if len(paths) < 2:
            raise HTTPException(400, "至少上传 2 张图片")
        source = {"kind": "images", "paths": paths, "upload_id": uid,
                  "count": len(paths)}
    elif kind == "video":
        uid = src.get("upload_id")
        if not uid or not re.fullmatch(r"[a-f0-9]{8,32}", uid):
            raise HTTPException(400, "upload_id 非法")
        d = (UPLOADS_DIR / uid).resolve()
        if not _is_under(d, UPLOADS_DIR) or not d.is_dir():
            raise HTTPException(404, "上传的视频不存在")
        videos = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in
                  {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}]
        if not videos:
            raise HTTPException(404, "上传的视频不存在")
        indices = sorted(set(int(i) for i in (src.get("indices") or [])))
        if len(indices) < 2:
            raise HTTPException(400, "至少选择 2 个视频帧")
        if len(indices) > MAX_FRAMES:
            raise HTTPException(400, f"最多选择 {MAX_FRAMES} 个视频帧")
        if min(indices) < 0 or max(indices) > 10_000_000:
            raise HTTPException(400, "视频帧索引超出范围")
        source = {"kind": "video", "video": str(videos[0]), "indices": indices,
                  "upload_id": uid}
    else:
        raise HTTPException(400, f"未知的 source.kind: {kind}")

    # ---- params
    clean_params = clean_recon_params(params)

    request = {
        "label": label,
        "source": source,
        "params": clean_params,
        "ckpt": CKPT,
        "backbone": BACKBONE,
        "model_label": "v5",
        "work_dir": str(WORK_DIR / "extract"),
    }
    job = create_job(request)
    return {"job_id": job["id"], "clip_id": job["clip_id"], "status": job["status"],
            "label": label}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return job_view(job)


@app.get("/api/jobs")
def api_jobs():
    return {"jobs": [job_view(j) for j in sorted(JOBS.values(), key=lambda x: -x["created"])[:20]]}


@app.post("/api/jobs/{job_id}/cancel")
def api_job_cancel(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    with JOB_LOCK:
        job["cancel_requested"] = True
        if job["status"] == "queued":
            job["status"] = "canceled"
            job["msg"] = "已取消"
        elif job["status"] == "running" and job.get("pid"):
            try:
                os.kill(job["pid"], 15)
            except Exception:
                pass
    return {"ok": True, "status": job["status"]}


@app.exception_handler(HTTPException)
async def http_exc_handler(request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _start_worker():
    t = threading.Thread(target=_worker_loop, daemon=True, name="recon-worker")
    t.start()


def _cleanup_incomplete_clips():
    for d in CLIPS_DIR.iterdir():
        if d.is_dir() and not (d / "meta.json").exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"[SERVER] removed incomplete clip dir: {d.name}", flush=True)


@app.on_event("startup")
def _startup():
    _cleanup_incomplete_clips()
    _start_worker()


def main():
    global GPU, CKPT, BACKBONE, PYTHON
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--gpu", default=os.environ.get("DEMO_GPU", "auto"),
                    help="auto=按实际显存/利用率选卡；也可写死物理编号，如 2")
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--backbone", default=str(BACKBONE))
    ap.add_argument("--python", default=PYTHON)
    ap.add_argument("--allowed-roots", default=os.environ.get("DEMO_ALLOWED_ROOTS", ""))
    args = ap.parse_args()

    GPU = str(args.gpu)
    CKPT = args.ckpt
    BACKBONE = args.backbone
    PYTHON = args.python
    if args.allowed_roots:
        roots = [Path(p).resolve() for p in args.allowed_roots.split(":") if p]
        ALLOWED_ROOTS[:] = roots
    print(f"[SERVER] data={DATA_DIR} gpu={GPU} python={PYTHON}", flush=True)
    print(f"[SERVER] http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
