#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render a pragmatic tech-style showcase video for the UAV reconstruction demo.

The heavy 3D pass is done by a small Three.js stage (stage.html/stage.js) driven
by Playwright; this script owns the timeline, camera path, projected object
detections, HUD and final ffmpeg encode.

Run with the Playwright-enabled env:
    cd /data/luhangze/aic/code_demo
    /data/luhangze/miniconda3/envs/modelscope/bin/python render_video.py

Preview a few key frames without encoding:
    ... render_video.py --preview 1.5,7,13.5,17,20,25,29,32
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
sys.path.insert(0, str(WORKSPACE))
from code_demo.objects import ClipData, detect_hybrid, draw_detections  # noqa: E402

W, H = 1920, 1080
FPS = 30
DURATION = 34.0

# ---------------------------------------------------------------- palette/fonts
PALETTE = json.loads((HERE / "palette.json").read_text())
NAMES = PALETTE["names"]
HEX = PALETTE["colors"]
RGB = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in HEX]
BG = (6, 10, 16)
TEXT = (232, 241, 251)
DIM = (125, 147, 170)
FAINT = (78, 96, 116)
CYAN = (56, 189, 248)
AMBER = (255, 209, 102)
GREEN = (52, 211, 153)
LINE = (32, 49, 67)

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

_fonts = {}


def F(path, size):
    key = (path, size)
    if key not in _fonts:
        try:
            _fonts[key] = ImageFont.truetype(path, size)
        except Exception:
            _fonts[key] = ImageFont.load_default()
    return _fonts[key]


def CJK(size):
    try:
        key = ("cjk", size)
        if key not in _fonts:
            _fonts[key] = ImageFont.truetype(FONT_CJK, size, index=0)
        return _fonts[key]
    except Exception:
        return F(FONT_REG, size)


# ------------------------------------------------------------------ utilities
def clamp(v, a, b):
    return max(a, min(b, v))


def smooth(x):
    x = clamp(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def mix(a, b, t):
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


def text(d, xy, s, font, fill, shadow=True, anchor=None):
    x, y = xy
    if shadow:
        a = fill[3] if len(fill) == 4 else 255
        d.text((x + 2, y + 2), s, font=font, fill=(0, 0, 0, int(a * 0.55)),
               anchor=anchor)
    d.text((x, y), s, font=font, fill=fill, anchor=anchor)


def tracked(d, xy, s, font, fill, tracking=2.0, shadow=False):
    x, y = xy
    for ch in s:
        if shadow:
            d.text((x + 1, y + 1), ch, font=font, fill=(0, 0, 0, 140))
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + tracking


def panel(d, box, radius=6, fill=(7, 12, 19, 200), outline=(38, 57, 77, 210)):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=1)


def dot(d, xy, r, fill):
    x, y = xy
    d.ellipse([x - r, y - r, x + r, y + r], fill=fill)


# ------------------------------------------------------------------ background
def make_hud_bg():
    """Top/bottom readability gradients + a subtle vignette."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = np.zeros((H, W), np.float32)
    # vignette
    cx, cy = W * 0.5, H * 0.5
    r = np.sqrt(((xx - cx) / (W * 0.62)) ** 2 + ((yy - cy) / (H * 0.62)) ** 2)
    a = np.maximum(a, np.clip((r - 0.55) / 0.55, 0, 1) * 92)
    # top gradient
    top = np.clip(1 - yy / 132.0, 0, 1) ** 1.5 * 185
    a = np.maximum(a, top)
    # bottom gradient
    bot = np.clip((yy - (H - 190)) / 190.0, 0, 1) ** 1.4 * 205
    a = np.maximum(a, bot)
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., 3] = a.astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


HUD_BG = make_hud_bg()


def make_left_grad():
    x = np.arange(W, dtype=np.float32)
    a = np.clip(1.0 - x / 1180.0, 0, 1) ** 1.15 * 190
    arr = np.zeros((H, W, 4), np.uint8)
    arr[..., 3] = np.tile(a.astype(np.uint8), (H, 1))
    return Image.fromarray(arr, "RGBA")


LEFT_GRAD = make_left_grad()


# ------------------------------------------------------------------ clip/model
class Show:
    def __init__(self, clip_name="city2-demo"):
        self.c = ClipData(clip_name)
        self.meta = self.c.meta
        self.route = np.asarray(self.meta["route"], np.float32)
        self.cams = np.asarray([w["cam"] for w in self.meta["waypoints"]], np.float32)
        self.counts = np.asarray(self.meta["counts"], np.int64)
        self.prefix = np.r_[0, np.cumsum(self.counts)]
        self.K = int(self.meta["num_waypoints"])
        self.N = int(self.meta["num_points"])
        self.center = self.c.center.astype(np.float32)
        self.radius = float(self.c.radius)
        self.lo = self.c.lo
        self.hi = self.c.hi
        self.dets = []
        self.obj_cam = None

    def state(self, p):
        p = clamp(p, 0, self.K - 1)
        k = int(math.floor(p + 1e-6))
        f = p - k
        vis = int(self.prefix[k + 1])
        if k + 1 < self.K:
            vis += int(round(f * self.counts[k + 1]))
        new = int(self.counts[k + 1]) if k + 1 < self.K else 0
        return k, f, vis, new

    def route_at(self, p):
        t = clamp(p / max(1, self.K - 1), 0, 1)
        f = t * (len(self.route) - 1)
        i0 = int(math.floor(f)); i1 = min(len(self.route) - 1, i0 + 1); a = f - i0
        return self.route[i0] * (1 - a) + self.route[i1] * a


def orbit(center, radius, elev, azim, dist):
    e = math.radians(elev); a = math.radians(azim)
    off = np.array([radius * dist * math.cos(e) * math.cos(a),
                    radius * dist * math.sin(e),
                    radius * dist * math.cos(e) * math.sin(a)], np.float32)
    return (np.asarray(center, np.float32) + off).tolist(), np.asarray(center, np.float32).tolist()


def camera_for(show: Show, shot, t, progress):
    c, r = show.center, show.radius
    if shot == "intro":
        u = clamp(t / 3.2, 0, 1)
        return orbit(c, r, 25, -64 + 5 * u, 2.82)
    if shot == "incremental":
        u = clamp((t - 3.2) / 9.4, 0, 1)
        k = int(clamp(progress, 0, show.K - 1))
        mean_cam = show.cams[:k + 1].mean(0)
        target = c * 0.55 + mean_cam * 0.45
        pos, tgt = orbit(target, r, 24, -60 + 11 * u, 2.62 - 0.04 * u)
        return pos, tgt
    if shot == "complete":
        u = clamp((t - 12.6) / 2.0, 0, 1)
        return orbit(c, r, 22, -50 + 3 * u, 2.78 + 0.10 * u)
    if shot == "split":
        u = clamp((t - 14.6) / 7.4, 0, 1)
        return orbit(c, r, 22, -62 + 16 * u, 2.80 - 0.04 * u)
    if shot == "objects":
        if show.obj_cam is None:
            show.obj_cam = orbit(c, r, 21, -56, 2.55)
        return show.obj_cam
    if shot == "outro":
        u = clamp((t - 30.0) / 4.0, 0, 1)
        return orbit(c, r, 26, -54 + 8 * u, 2.80)
    return orbit(c, r, 22, -56, 2.1)


def shot_at(t):
    if t < 3.2:
        return "intro"
    if t < 12.6:
        return "incremental"
    if t < 14.6:
        return "complete"
    if t < 22.0:
        return "split"
    if t < 30.0:
        return "objects"
    return "outro"


def progress_at(t):
    if t < 3.2:
        return 1.2 * smooth(t / 3.2)
    if t < 12.6:
        u = (t - 3.2) / 9.4
        return 1.2 + (15.0 - 1.2) * (0.35 * u + 0.65 * smooth(u))
    return 15.0


def split_at(t):
    if t < 14.6:
        return None
    if t < 22.0:
        u = smooth(clamp((t - 14.8) / 2.4, 0, 1))
        return 0.20 + 0.30 * u
    if t < 30.0:
        return 0.50
    return 0.50


def new_glow_at(t, progress, shot):
    if shot != "incremental":
        return 0.0
    fr = progress - math.floor(progress)
    return 0.12 * (math.sin(math.pi * fr) ** 2)


def build_plan(show: Show, fps, duration):
    n = int(round(fps * duration))
    plan = []
    for i in range(n):
        t = i / fps
        shot = shot_at(t)
        p = progress_at(t)
        k, f, vis, new = show.state(p)
        pos, tgt = camera_for(show, shot, t, p)
        split = split_at(t)
        stage = {
            "progress": p, "mode": "split" if split is not None and shot in ("split", "outro") else "rgb",
            "split": split if split is not None else 0.5,
            "camera": {"pos": pos, "target": tgt, "fov": 45},
            "pointSize": 4.0, "newGlow": new_glow_at(t, p, shot),
            "showRoute": shot in ("intro", "incremental", "complete"),
            "showGrid": shot not in ("objects",),
        }
        plan.append({"i": i, "t": t, "shot": shot, "progress": p, "k": k, "frac": f,
                     "visible": vis, "new": new, "split": split, "stage": stage})
    return plan


# ------------------------------------------------------------------ detections
def prepare_detections(show: Show):
    pos, tgt = orbit(show.center, show.radius, 21, -56, 2.55)
    show.obj_cam = (pos, tgt)
    t0 = time.time()
    dets = detect_hybrid(show.c, pos, tgt, 45, W, H)
    # merge/drop near-duplicate small objects; keep at most 7
    keep = []
    for d in dets:
        if d["cls"] in (9, 10) and any(
                q["cls"] == d["cls"] and _box_center_dist(q["box"], d["box"]) < 45 for q in keep):
            continue
        keep.append(d)
    show.dets = keep[:7]
    print(f"[DETECT] {len(show.dets)} objects in {time.time()-t0:.1f}s")
    for d in show.dets:
        print(f"  {d['name']:8s} {d['conf']:.2f} box={[round(v) for v in d['box']]}")


def _box_center_dist(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    return math.hypot((ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2))


# ------------------------------------------------------------------ HUD drawing
def draw_top(ov, show: Show, t):
    d = ImageDraw.Draw(ov, "RGBA")
    # logo mark
    cx, cy = 62, 48
    d.polygon([(cx, cy - 11), (cx + 11, cy), (cx, cy + 11), (cx - 11, cy)],
              outline=CYAN + (235,), width=2)
    d.polygon([(cx, cy - 4), (cx + 4, cy), (cx, cy + 4), (cx - 4, cy)], fill=CYAN + (230,))
    text(d, (84, 30), "AIC", F(FONT_BOLD, 25), TEXT + (255,))
    tracked(d, (85, 62), "SEGSTREAMVGGT / URBAN RECONSTRUCTION", F(FONT_MONO, 11),
            DIM + (225,), tracking=1.1)
    d.line([(60, 88), (560, 88)], fill=(42, 62, 82, 160), width=1)

    # right-side model chips (fit within the frame)
    items = [("MODEL", "v5"), ("GEOMETRY", "DEPTH+POSE"),
             ("WAYPOINTS", str(show.K)), ("POINTS", f"{show.N/1e6:.1f}M")]
    widths = []
    for lab, val in items:
        widths.append(max(d.textlength(val, font=F(FONT_MONO, 14)),
                          d.textlength(lab, font=F(FONT_MONO, 9))) + 26)
    x = W - 60 - sum(widths) + 26
    for j, (lab, val) in enumerate(items):
        text(d, (x, 38), val, F(FONT_MONO, 14), TEXT + (245,))
        text(d, (x, 58), lab, F(FONT_MONO, 9), FAINT + (235,))
        x += widths[j]
        if j < len(items) - 1:
            d.line([(x - 13, 38), (x - 13, 67)], fill=(45, 64, 84, 170), width=1)
    ready = "MODEL READY"
    rw = d.textlength(ready, font=F(FONT_MONO, 9))
    text(d, (W - 60, 76), ready, F(FONT_MONO, 9), GREEN + (220,), anchor="ra")
    dot(d, (W - 60 - rw - 10, 84), 3, GREEN + (255,))


def draw_stage_title(ov, show: Show, t, shot, p, vis, new):
    d = ImageDraw.Draw(ov, "RGBA")
    tags = {
        "incremental": ("01", "FLIGHT-SYNCED RECONSTRUCTION", "INCREMENTAL UAV POINT CLOUD"),
        "complete": ("16", "FULL SCENE RECONSTRUCTION", "ALL 16 WAYPOINTS MERGED"),
        "split": ("02", "RGB ↔ SEMANTIC", "ONE POINT CLOUD · LEFT RGB · RIGHT SEMANTIC"),
        "objects": ("03", "OBJECT-AWARE RECONSTRUCTION", "PROJECTED SEMANTIC DETECTIONS"),
    }
    if shot not in tags:
        return
    tag, title, sub = tags[shot]
    d.rectangle([60, 900, 65, 956], fill=CYAN + (235,))
    text(d, (80, 896), f"{tag}   {title}", F(FONT_BOLD, 21), TEXT + (250,))
    text(d, (81, 928), sub, F(FONT_MONO, 10), DIM + (230,))
    if shot == "incremental":
        stat = f"WAYPOINT {min(show.K, int(math.floor(p)) + 1):02d} / {show.K:02d}    ·    {vis:,} PTS"
        if new:
            stat += f"    ·    +{new:,} NEW"
    elif shot == "objects":
        uniq = len(set(x["cls"] for x in show.dets))
        stat = f"{len(show.dets)} OBJECTS    ·    {uniq} CLASSES    ·    {show.N:,} PTS"
    else:
        stat = f"{show.N:,} PTS    ·    11 SEMANTIC CLASSES"
    text(d, (81, 952), stat, F(FONT_MONO, 12), (172, 192, 212, 240))
    if shot == "incremental":
        # live indicator
        blink = 0.55 + 0.45 * math.sin(t * 5.0)
        label = "RECONSTRUCTING"
        lw = d.textlength(label, font=F(FONT_MONO, 9))
        text(d, (W - 60, 900), label, F(FONT_MONO, 9), DIM + (220,), anchor="ra")
        dot(d, (W - 60 - lw - 10, 908), 3.5, (255, 92, 92, int(210 * blink) + 40))


def draw_timeline(ov, show: Show, t, p, shot):
    if shot in ("intro", "outro"):
        return
    d = ImageDraw.Draw(ov, "RGBA")
    x0, x1, y = 80, W - 80, 1016
    d.line([(x0, y), (x1, y)], fill=(44, 63, 82, 220), width=2)
    frac = clamp(p / max(1, show.K - 1), 0, 1)
    xc = x0 + (x1 - x0) * frac
    d.line([(x0, y), (xc, y)], fill=CYAN + (235,), width=2)
    for i in range(show.K):
        x = x0 + (x1 - x0) * i / (show.K - 1)
        on = x <= xc + 1
        d.line([(x, y - 5), (x, y + 5)], fill=(CYAN if on else (60, 82, 104)) + (220,), width=2)
    dot(d, (xc, y), 5, AMBER + (255,))
    d.ellipse([xc - 9, y - 9, xc + 9, y + 9], outline=AMBER + (100,), width=2)
    wp = min(show.K, int(math.floor(p)) + 1)
    text(d, (x0, y + 12), "01", F(FONT_MONO, 9), FAINT + (220,))
    text(d, (x1, y + 12), f"{show.K:02d}", F(FONT_MONO, 9), FAINT + (220,), anchor="ra")
    text(d, (xc, y + 12), f"{wp:02d}", F(FONT_MONO, 10), AMBER + (240,), anchor="ma")


def draw_minimap(ov, show: Show, p, t):
    d = ImageDraw.Draw(ov, "RGBA")
    x0, y0, w, h = W - 302, 118, 242, 168
    panel(d, [x0, y0, x0 + w, y0 + h], radius=7)
    text(d, (x0 + 12, y0 + 10), "FLIGHT ROUTE", F(FONT_MONO, 9), DIM + (235,))
    r = show.route
    mnx, mxx = float(r[:, 0].min()), float(r[:, 0].max())
    mnz, mxz = float(r[:, 2].min()), float(r[:, 2].max())
    pad = 18
    sc = min((w - 2 * pad) / max(1e-6, mxx - mnx), (h - 34 - 2 * pad) / max(1e-6, mxz - mnz))
    cx, cz = (mnx + mxx) / 2, (mnz + mxz) / 2
    def px(x): return x0 + w / 2 + (x - cx) * sc
    def py(z): return y0 + 26 + (h - 26) / 2 + (z - cz) * sc
    pts = [(px(v[0]), py(v[2])) for v in r]
    d.line(pts, fill=(55, 74, 94, 220), width=2, joint="curve")
    frac = clamp(p / max(1, show.K - 1), 0, 1)
    n = max(1, int(round(frac * (len(pts) - 1))) + 1)
    d.line(pts[:n], fill=CYAN + (220,), width=2, joint="curve")
    for i, wpt in enumerate(show.meta["waypoints"]):
        if i % 4 == 0 or i == show.K - 1:
            d.ellipse([px(wpt["cam"][0]) - 1.6, py(wpt["cam"][2]) - 1.6,
                       px(wpt["cam"][0]) + 1.6, py(wpt["cam"][2]) + 1.6],
                      fill=(110, 130, 150, 210))
    cp = show.route_at(p)
    d.ellipse([px(cp[0]) - 4, py(cp[2]) - 4, px(cp[0]) + 4, py(cp[2]) + 4], fill=AMBER + (255,))
    d.ellipse([px(cp[0]) - 8, py(cp[2]) - 8, px(cp[0]) + 8, py(cp[2]) + 8], outline=AMBER + (90,), width=2)


def draw_legend(ov, show: Show, t, shot):
    if shot not in ("split", "objects"):
        return
    d = ImageDraw.Draw(ov, "RGBA")
    x0, y0 = W - 480, 900
    w, h = 420, 64
    panel(d, [x0, y0, x0 + w, y0 + h], radius=7)
    if shot == "split":
        items = [(1, "BUILDING"), (2, "ROAD"), (5, "TREE"), (7, "TERRAIN"),
                 (9, "VEHICLE"), (10, "PERSON")]
        cols = 3
    else:
        counts = {}
        for x in show.dets:
            counts[x["cls"]] = counts.get(x["cls"], 0) + 1
        items = [(c, f"{NAMES[c].upper()} ×{n}") for c, n in counts.items()]
        cols = min(3, max(1, len(items)))
    for i, (c, label) in enumerate(items):
        cx = x0 + 14 + (i % cols) * 132
        cy = y0 + 16 + (i // cols) * 24
        d.rounded_rectangle([cx, cy, cx + 9, cy + 9], radius=2, fill=RGB[c] + (255,))
        text(d, (cx + 15, cy - 3), label, F(FONT_MONO, 10), TEXT + (235,))


def draw_intro(ov, t):
    a = smooth(clamp((t - 0.25) / 0.7, 0, 1)) * (1 - smooth(clamp((t - 2.75) / 0.45, 0, 1)))
    if a <= 0:
        return
    d = ImageDraw.Draw(ov, "RGBA")
    x, y = 96, 330
    d.rectangle([x, y, x + 4, y + 232], fill=CYAN + (int(230 * a),))
    tracked(d, (x + 24, y - 6), "AIC  /  SEGSTREAMVGGT", F(FONT_MONO, 12),
            CYAN + (int(235 * a),), tracking=2)
    text(d, (x + 24, y + 28), "URBAN", F(FONT_BOLD, 78), TEXT + (int(255 * a),))
    text(d, (x + 24, y + 106), "RECONSTRUCTION", F(FONT_BOLD, 78), TEXT + (int(255 * a),))
    d.line([(x + 26, y + 202), (x + 720, y + 202)], fill=(56, 189, 248, int(170 * a)), width=2)
    text(d, (x + 26, y + 216), "RGB + SEMANTIC 3D POINT CLOUD", F(FONT_MONO, 19),
         (210, 226, 240, int(240 * a)))
    text(d, (x + 26, y + 246), "无人机城市重建与语义分割", CJK(20),
         (150, 176, 198, int(230 * a)))
    # small chips
    chips = ["16 WAYPOINTS", "1.5M POINTS", "11 CLASSES"]
    cx = x + 26
    for s in chips:
        tw = d.textlength(s, font=F(FONT_MONO, 10))
        d.rounded_rectangle([cx, y + 282, cx + tw + 18, y + 304], radius=4,
                            fill=(12, 20, 30, int(190 * a)), outline=(44, 65, 86, int(200 * a)))
        text(d, (cx + 9, y + 288), s, F(FONT_MONO, 10), DIM + (int(230 * a),))
        cx += tw + 28


def draw_complete(ov, t):
    a = smooth(clamp((t - 12.65) / 0.35, 0, 1)) * (1 - smooth(clamp((t - 14.05) / 0.4, 0, 1)))
    if a <= 0:
        return
    d = ImageDraw.Draw(ov, "RGBA")
    cx, y = W / 2, 106
    panel(d, [cx - 360, y, cx + 360, y + 66], radius=7,
          fill=(6, 11, 18, int(198 * a)), outline=(45, 67, 89, int(200 * a)))
    text(d, (cx, y + 5), "RECONSTRUCTION COMPLETE", F(FONT_BOLD, 28), TEXT + (int(252 * a),), anchor="ma")
    text(d, (cx, y + 42), "16 / 16 WAYPOINTS   ·   1,500,000 POINTS", F(FONT_MONO, 13),
         CYAN + (int(240 * a),), anchor="ma")


def draw_outro(ov, t):
    a = smooth(clamp((t - 30.15) / 0.7, 0, 1)) * (1 - smooth(clamp((t - 33.15) / 0.65, 0, 1)))
    if a <= 0:
        return
    d = ImageDraw.Draw(ov, "RGBA")
    x, y = 96, 330
    d.rectangle([x, y, x + 4, y + 214], fill=CYAN + (int(230 * a),))
    tracked(d, (x + 24, y - 6), "AIC  /  UAV RECONSTRUCTION", F(FONT_MONO, 12),
            CYAN + (int(235 * a),), tracking=2)
    text(d, (x + 24, y + 30), "RECONSTRUCTION", F(FONT_BOLD, 62), TEXT + (int(255 * a),))
    text(d, (x + 24, y + 98), "COMPLETE", F(FONT_BOLD, 62), TEXT + (int(255 * a),))
    d.line([(x + 26, y + 176), (x + 640, y + 176)], fill=(56, 189, 248, int(170 * a)), width=2)
    text(d, (x + 26, y + 190), "16 WAYPOINTS  ·  1.5M POINTS  ·  11 CLASSES", F(FONT_MONO, 15),
         (205, 222, 238, int(240 * a)))
    text(d, (x + 26, y + 220), "SegStreamVGGT v5  ·  depth + pose  ·  query segmentation",
         F(FONT_MONO, 11), DIM + (int(230 * a),))


def draw_split(ov, show: Show, t, split, labels=True):
    if split is None:
        return
    d = ImageDraw.Draw(ov, "RGBA")
    x = int(split * W)
    # soft glow + crisp line
    for wdt, aa in ((14, 22), (6, 55), (2, 220)):
        d.line([(x, 0), (x, H)], fill=(56, 189, 248, aa), width=wdt)
    d.line([(x, 96), (x, 986)], fill=(230, 246, 255, 240), width=2)
    # ticks / handle
    for yy in (96, 986):
        d.line([(x - 10, yy), (x + 10, yy)], fill=(230, 246, 255, 220), width=2)
    d.polygon([(x, 528), (x + 7, 540), (x, 552), (x - 7, 540)], outline=(235, 249, 255, 240), width=2)
    if not labels:
        return
    # labels
    lx = x - 24
    text(d, (lx, 112), "RGB", F(FONT_BOLD, 18), (245, 250, 255, 250), anchor="ra")
    text(d, (lx, 138), "POINT CLOUD", F(FONT_MONO, 9), (173, 193, 212, 235), anchor="ra")
    rx = x + 24
    text(d, (rx, 112), "SEMANTIC", F(FONT_BOLD, 18), CYAN + (255,))
    text(d, (rx, 138), "POINT CLOUD", F(FONT_MONO, 9), (150, 205, 235, 235))
    # mode chip
    cw = d.textlength("SPLIT VIEW", font=F(FONT_MONO, 9))
    d.rounded_rectangle([x - cw / 2 - 8, 64, x + cw / 2 + 8, 82], radius=4,
                        fill=(8, 15, 23, 210), outline=(46, 67, 88, 210))
    text(d, (x - cw / 2, 69), "SPLIT VIEW", F(FONT_MONO, 9), DIM + (235,))


def draw_fade(ov, alpha):
    if alpha <= 0:
        return
    d = ImageDraw.Draw(ov, "RGBA")
    d.rectangle([0, 0, W, H], fill=(0, 0, 0, int(255 * clamp(alpha, 0, 1))))


def global_fade(t):
    a = 1.0
    if t < 0.55:
        a = min(a, smooth(t / 0.55))
    if t > 33.25:
        a = min(a, smooth((34.0 - t) / 0.75))
    for b in (3.2, 12.6, 14.6, 22.0, 30.0):
        if abs(t - b) < 0.12:
            a = min(a, abs(t - b) / 0.12)
    return a


# ------------------------------------------------------------------ compose one
def compose_frame(raw_path, plan_item, show: Show, out_path):
    t = plan_item["t"]; shot = plan_item["shot"]; p = plan_item["progress"]
    img = Image.open(raw_path).convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(0.94)
    img = ImageEnhance.Contrast(img).enhance(1.14)
    img = ImageEnhance.Color(img).enhance(1.12)
    base = Image.alpha_composite(img.convert("RGBA"), HUD_BG)
    if shot in ("intro", "outro"):
        base = Image.alpha_composite(base, LEFT_GRAD)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # scene-space split divider labels and detections
    if shot in ("split", "outro"):
        draw_split(ov, show, t, plan_item["split"] if plan_item["split"] is not None else 0.5,
                   labels=(shot == "split"))
    if shot == "objects":
        dt = (t - 22.0) + 0.10
        draw_detections(ov, show.dets,
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", t=dt)

    # HUD
    draw_top(ov, show, t)
    if shot == "intro":
        draw_intro(ov, t)
    elif shot == "complete":
        draw_complete(ov, t)
        draw_stage_title(ov, show, t, shot, p, plan_item["visible"], plan_item["new"])
    elif shot == "outro":
        draw_outro(ov, t)
    else:
        draw_stage_title(ov, show, t, shot, p, plan_item["visible"], plan_item["new"])

    if shot in ("intro", "incremental", "complete"):
        draw_minimap(ov, show, p, t)
    draw_legend(ov, show, t, shot)
    draw_timeline(ov, show, t, p, shot)
    draw_fade(ov, 1.0 - global_fade(t))

    out = Image.alpha_composite(base, ov).convert("RGB")
    out.save(out_path, quality=95)
    return out


# ------------------------------------------------------------------ HTTP / capture
class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def start_server():
    handler = partial(Quiet, directory=str(WORKSPACE))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    return httpd, port


def capture(plan, raw_dir, clip_name, resume=False):
    from playwright.sync_api import sync_playwright
    httpd, port = start_server()
    print(f"[HTTP] serving {WORKSPACE} on 127.0.0.1:{port}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(
                channel="msedge", headless=True,
                args=["--no-sandbox", "--enable-unsafe-swiftshader",
                      "--disable-dev-shm-usage", "--ignore-gpu-blocklist"])
            pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
            pg.goto(f"http://127.0.0.1:{port}/code_demo/stage.html?clip={clip_name}",
                    wait_until="domcontentloaded", timeout=120000)
            pg.wait_for_function("window.__ready===true || window.__error", timeout=180000)
            if pg.evaluate("window.__error || null"):
                raise RuntimeError(pg.evaluate("window.__error"))
            pg.wait_for_timeout(2500)
            # warm-up frame + one extra render to get past the initial context loss
            pg.evaluate("f=>window.setFrame(f)", plan[0]["stage"])
            pg.wait_for_timeout(700)
            pg.evaluate("f=>window.setFrame(f)", plan[0]["stage"])
            pg.wait_for_timeout(200)
            n = len(plan)
            for i, item in enumerate(plan):
                path = raw_dir / f"raw_{i:05d}.jpg"
                if resume and path.exists() and path.stat().st_size > 50000:
                    continue
                pg.evaluate("f=>window.setFrame(f)", item["stage"])
                pg.wait_for_timeout(25)
                ok = False
                for attempt in range(3):
                    pg.screenshot(path=str(path), type="jpeg", quality=94)
                    if path.exists() and path.stat().st_size > 50000:
                        ok = True
                        break
                    pg.wait_for_timeout(800)
                    pg.evaluate("f=>window.setFrame(f)", item["stage"])
                if not ok:
                    print(f"[WARN] blank frame {i} ({item['t']:.2f}s)")
                if i % 30 == 0 or i == n - 1:
                    el = time.time() - t0
                    print(f"[CAPTURE] {i+1}/{n}  {el:.1f}s  ({(i+1)/max(el,1e-6):.2f} fps)", flush=True)
            b.close()
    finally:
        httpd.shutdown()
    print(f"[CAPTURE] done in {time.time()-t0:.1f}s -> {raw_dir}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="city2-demo")
    ap.add_argument("--out", default=str(HERE / "output" / "uav_reconstruction_demo.mp4"))
    ap.add_argument("--preview", default="", help="comma-separated timestamps; render only these")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-capture", action="store_true")
    ap.add_argument("--duration", type=float, default=DURATION)
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()

    show = Show(args.clip)
    print(f"[CLIP] {args.clip}: {show.N:,} pts, {show.K} waypoints, center={show.center}, r={show.radius:.3f}")
    prepare_detections(show)

    if args.preview:
        times = [float(x) for x in args.preview.split(",") if x.strip()]
        plan = []
        for i, t in enumerate(times):
            shot = shot_at(t); p = progress_at(t)
            k, f, vis, new = show.state(p)
            pos, tgt = camera_for(show, shot, t, p)
            split = split_at(t)
            plan.append({"i": i, "t": t, "shot": shot, "progress": p, "k": k, "frac": f,
                         "visible": vis, "new": new, "split": split,
                         "stage": {"progress": p,
                                   "mode": "split" if split is not None and shot in ("split", "outro") else "rgb",
                                   "split": split if split is not None else 0.5,
                                   "camera": {"pos": pos, "target": tgt, "fov": 45},
                                   "pointSize": 4.0, "newGlow": new_glow_at(t, p, shot),
                                   "showRoute": shot in ("intro", "incremental", "complete"),
                                   "showGrid": shot not in ("objects",)}})
        raw_dir = HERE / "frames" / "preview_raw"
        capture(plan, raw_dir, args.clip, resume=False)
        pdir = HERE / "preview"; pdir.mkdir(exist_ok=True)
        for i, item in enumerate(plan):
            out = pdir / f"frame_{item['t']:05.1f}s_{item['shot']}.jpg"
            compose_frame(raw_dir / f"raw_{i:05d}.jpg", item, show, out)
            print(f"[PREVIEW] {out}")
        return

    plan = build_plan(show, args.fps, args.duration)
    plan_path = HERE / "frames" / "plan.json"
    plan_path.parent.mkdir(exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")

    raw_dir = HERE / "frames" / "raw"
    if not args.skip_capture:
        capture(plan, raw_dir, args.clip, resume=args.resume)

    final_dir = HERE / "frames" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"[COMPOSE] {len(plan)} frames -> {final_dir}")
    t0 = time.time()
    for i, item in enumerate(plan):
        compose_frame(raw_dir / f"raw_{i:05d}.jpg", item, show, final_dir / f"final_{i:05d}.jpg")
        if i % 60 == 0:
            print(f"[COMPOSE] {i+1}/{len(plan)}  {time.time()-t0:.1f}s", flush=True)
    print(f"[COMPOSE] done in {time.time()-t0:.1f}s")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(final_dir / "final_%05d.jpg"),
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    print("[ENCODE]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[DONE] {out}  ({out.stat().st_size/1e6:.1f} MB)")

    # poster from a representative frame
    poster_i = min(len(plan) - 1, int(round(26.0 * args.fps)))
    shutil.copyfile(final_dir / f"final_{poster_i:05d}.jpg", HERE / "output" / "poster.jpg")


if __name__ == "__main__":
    main()
