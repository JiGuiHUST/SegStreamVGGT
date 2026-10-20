"""Projected point-cloud object detection for the showcase video.

We do not have 2D annotations for the UAV frames, so we derive YOLO-style
screen-space boxes/contours from the reconstructed semantic point cloud:
project points with the same camera as the Three.js stage, rasterise per-class
masks, take connected components, then return boxes, simplified contours,
class names and a confidence (mean semantic probability).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from make_demo import core

HERE = Path(__file__).resolve().parent
CLIP_DIR = HERE.parent / "make_demo" / "data" / "clips"


class ClipData:
    def __init__(self, clip: str = "city2-demo"):
        self.clip = clip
        self.dir = CLIP_DIR / clip
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        d = core.load_point_buffer(self.dir / "points.bin")
        self.points = d["points"].astype(np.float32)
        self.rgb = d["rgb"]
        self.cls = d["cls"].astype(np.int32)
        self.wp = d["wp"].astype(np.int32)
        self.prob = d["prob"].astype(np.float32) / 255.0
        self.palette = json.loads((HERE / "palette.json").read_text())
        self.class_names = self.palette["names"]
        lo = np.array(self.meta["bounds"][0], np.float32)
        hi = np.array(self.meta["bounds"][1], np.float32)
        self.center = (lo + hi) * 0.5
        self.radius = float(np.linalg.norm(hi - lo) * 0.5)
        self.lo, self.hi = lo, hi
        self.N = len(self.points)
        self.K = int(self.meta["num_waypoints"])


def view_matrix(pos, target, up=(0.0, 1.0, 0.0)):
    pos = np.asarray(pos, np.float64)
    target = np.asarray(target, np.float64)
    z = pos - target
    z /= np.linalg.norm(z) + 1e-12
    up = np.asarray(up, np.float64)
    x = np.cross(up, z)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    t = -R @ pos
    return R, t


def project(points, pos, target, fov_deg, W, H, near=0.01):
    R, t = view_matrix(pos, target)
    pc = points @ R.T + t
    z = -pc[:, 2]
    good = z > near
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    aspect = W / H
    x = (f / aspect) * pc[:, 0] / z
    y = f * pc[:, 1] / z
    sx = (x * 0.5 + 0.5) * W
    sy = (0.5 - y * 0.5) * H
    good &= (sx >= -20) & (sx <= W + 20) & (sy >= -20) & (sy <= H + 20)
    return sx, sy, good


def _component_contours(mask, scale, min_area, top_n, max_area_frac=0.9):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cand = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if area > mask.shape[0] * mask.shape[1] * max_area_frac:
            continue
        cand.append((area, i, x, y, w, h))
    cand.sort(reverse=True)
    out = []
    for area, i, x, y, w, h in cand[:top_n]:
        sub = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        eps = 2.0
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
        c = c / scale
        out.append({
            "mask_index": i,
            "box": [float(x) / scale, float(y) / scale, float(w) / scale, float(h) / scale],
            "area": float(area) / (scale * scale),
            "contour": c.round(1).tolist(),
        })
    return out, labels


def detect(clip: ClipData, pos, target, fov_deg, W, H, scale=0.5,
           classes=(1, 5, 9, 10), min_points=0, progress=None):
    """Return list of detections in full-resolution screen coordinates."""
    p = clip.points
    if progress is not None:
        # use points up to the current reveal (all for the object shot)
        k = int(math.floor(progress + 1e-6))
        f = progress - k
        use = (clip.wp <= k) | ((clip.wp == k + 1) & (clip.rank <= f))
        p = p[use]
        cls = clip.cls[use]
        prob = clip.prob[use]
    else:
        cls = clip.cls
        prob = clip.prob
    sx, sy, good = project(p, pos, target, fov_deg, W, H)
    xs = np.clip((sx * scale).astype(np.int32), 0, int(W * scale) - 1)
    ys = np.clip((sy * scale).astype(np.int32), 0, int(H * scale) - 1)
    good &= (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    xs, ys, cls, prob = xs[good], ys[good], cls[good], prob[good]
    H2, W2 = int(H * scale), int(W * scale)

    specs = {
        1: {"min_area": max(300, int(0.0012 * W2 * H2)), "top": 3, "name": "BUILDING"},
        5: {"min_area": max(30, int(0.00008 * W2 * H2)), "top": 4, "name": "TREE"},
        9: {"min_area": max(8, int(0.000015 * W2 * H2)), "top": 3, "name": "VEHICLE"},
        10: {"min_area": max(6, int(0.000012 * W2 * H2)), "top": 2, "name": "PERSON"},
    }
    dets = []
    for c in classes:
        m = cls == c
        if not m.any():
            continue
        mask = np.zeros((H2, W2), np.uint8)
        mask[ys[m], xs[m]] = 255
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        spec = specs[c]
        comps, labels = _component_contours(mask, scale, spec["min_area"], spec["top"])
        # confidence per component
        comp_ids = labels[ys[m], xs[m]]
        sums = np.bincount(comp_ids, weights=prob[m], minlength=labels.max() + 1)
        cnts = np.bincount(comp_ids, minlength=labels.max() + 1)
        for d in comps:
            i = d["mask_index"]
            conf = float(sums[i] / max(1, cnts[i]))
            d.update({"cls": c, "name": spec["name"], "conf": conf,
                      "color": clip.palette["colors"][c], "n_points": int(cnts[i])})
            dets.append(d)
    # largest first, at most 10
    dets.sort(key=lambda d: d["area"], reverse=True)
    return dets[:10]


def draw_detections(img, dets, font_path, mono_path, t=1.0):
    """Draw YOLO-style boxes, contours, labels and leader lines on a PIL image."""
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    try:
        font = ImageFont.truetype(font_path, 21)
        font_small = ImageFont.truetype(mono_path, 15)
    except Exception:
        font = ImageFont.load_default(); font_small = font
    labels = []
    safe = (40, 116, W - 40, H - 104)
    for idx, d in enumerate(dets):
        alpha = max(0.0, min(1.0, (t - idx * 0.30) / 0.35))
        if alpha <= 0:
            continue
        col = d["color"].lstrip("#")
        rgb = tuple(int(col[i:i+2], 16) for i in (0, 2, 4))
        x, y, w, h = [int(v) for v in d["box"]]
        x = max(2, min(W - 4, x)); y = max(2, min(H - 4, y))
        w = max(3, min(W - x - 2, w)); h = max(3, min(H - y - 2, h))
        pts = [(int(px), int(py)) for px, py in d["contour"]]
        if len(pts) > 2:
            draw.line(pts + [pts[0]], fill=rgb + (int(170 * alpha),), width=2, joint="curve")
        draw.rectangle([x, y, x + w, y + h], outline=rgb + (int(215 * alpha),), width=2)
        L = min(24, max(9, int(min(w, h) * 0.20)))
        for (cx, cy, dx, dy) in [(x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)]:
            draw.line([(cx, cy), (cx + dx * L, cy)], fill=rgb + (int(255 * alpha),), width=4)
            draw.line([(cx, cy), (cx, cy + dy * L)], fill=rgb + (int(255 * alpha),), width=4)
        text = f"{d['name']}  {d['conf']:.2f}"
        bb = draw.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        cw, chh = tw + 18, th + 12
        options = [
            (x, y - chh - 8, "above"),
            (x + w - cw, y - chh - 8, "above"),
            (x + 7, y + 7, "inside"),
            (x, y + h + 8, "below"),
            (x + w - cw, y + h + 8, "below"),
        ]
        chosen = None
        for (lx, ly, pos) in options:
            lx = int(max(safe[0], min(safe[2] - cw, lx)))
            ly = int(max(safe[1], min(safe[3] - chh, ly)))
            rect = (lx, ly, lx + cw, ly + chh)
            if all(not (rect[0] < b[2] and rect[2] > b[0] and rect[1] < b[3] and rect[3] > b[1]) for b in labels):
                chosen = (lx, ly, pos, rect)
                break
        if chosen is None:
            lx = int(max(safe[0], min(safe[2] - cw, x)))
            ly = int(max(safe[1], min(safe[3] - chh, y - chh - 8)))
            chosen = (lx, ly, "above", (lx, ly, lx + cw, ly + chh))
        lx, ly, pos, rect = chosen
        labels.append(rect)
        draw.rounded_rectangle([lx, ly, lx + cw, ly + chh], radius=4,
                               fill=(8, 14, 22, int(232 * alpha)), outline=rgb + (int(210 * alpha),), width=1)
        draw.rectangle([lx, ly, lx + 4, ly + chh], fill=rgb + (int(255 * alpha),))
        draw.text((lx + 10, ly + 5), text, font=font, fill=(240, 247, 255, int(255 * alpha)))
        # small id / points chip on the box
        tag = f"{idx+1:02d}  {d['n_points']} pts"
        draw.text((x + 3, y + h + 4), tag, font=font_small, fill=rgb + (int(225 * alpha),),
                  stroke_width=2, stroke_fill=(6, 10, 16, 200))
        # leader line from the label to the nearest box corner (elbow)
        if pos == "above":
            sx0, sy0 = lx + cw / 2, ly + chh
            tx0, ty0 = x + min(w * 0.15, 40), y
            my = (sy0 + ty0) / 2
            draw.line([(sx0, sy0), (sx0, my), (tx0, my), (tx0, ty0)],
                      fill=rgb + (int(150 * alpha),), width=1)
    return img


def _cluster_contour(x, y, W, H, scale=0.5):
    """Rasterise a cluster's screen points and return a simplified contour."""
    H2, W2 = int(H * scale), int(W * scale)
    xs = np.clip((x * scale).astype(np.int32), 0, W2 - 1)
    ys = np.clip((y * scale).astype(np.int32), 0, H2 - 1)
    mask = np.zeros((H2, W2), np.uint8)
    mask[ys, xs] = 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2).astype(np.float32) / scale
    return c.round(1).tolist()




def _kmeans_labels(data, k, seed=0):
    """Small deterministic k-means (centroids on a sample, assign all with KDTree).

    Avoids scikit-learn so the renderer only needs numpy/scipy/cv2/PIL.
    """
    from scipy.cluster.vq import kmeans2
    from scipy.spatial import cKDTree

    data = np.asarray(data, np.float32)
    step = max(1, len(data) // 60000)
    sample = data[::step]
    try:
        centroids, _ = kmeans2(sample, k, iter=25, minit="++", seed=seed)
    except Exception:
        # fallback: quantile split along x
        q = np.quantile(data[:, 0], np.linspace(0, 1, k + 1)[1:-1])
        centroids = np.stack([q, np.full_like(q, np.median(data[:, 1]))], 1).astype(np.float32)
    if centroids.ndim == 1:
        centroids = centroids.reshape(1, -1)
    tree = cKDTree(centroids.astype(np.float64))
    _, labels = tree.query(data.astype(np.float64), k=1)
    return labels.astype(np.int32)


def detect_hybrid(clip: ClipData, pos, target, fov_deg, W, H,
                  building_k=5, tree_k=6, seed=0):
    """YOLO-style detections: KMeans regions for building/tree + components for
    vehicle/person.  Returns at most 8 stable detections for a fixed hero view."""
    sx, sy, good = project(clip.points, pos, target, fov_deg, W, H)
    good &= (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    dets = []

    for cls, k, max_w, max_h, top in (
        (1, building_k, 0.48, 0.94, 2),
        (5, tree_k, 0.30, 0.50, 3),
    ):
        m = (clip.cls == cls) & good
        p = clip.points[m]
        if len(p) < 100:
            continue
        lab = _kmeans_labels(p[:, [0, 2]], k, seed)
        cand = []
        for i in range(k):
            mm = lab == i
            n = int(mm.sum())
            if n < 200:
                continue
            x = sx[m][mm]; y = sy[m][mm]; pr = clip.prob[m][mm]
            x0, x1, y0, y1 = float(x.min()), float(x.max()), float(y.min()), float(y.max())
            bw, bh = x1 - x0, y1 - y0
            if bw < 30 or bh < 30 or bw > max_w * W or bh > max_h * H:
                continue
            contour = _cluster_contour(x, y, W, H)
            if not contour:
                continue
            ca = np.asarray(contour, np.float32)
            x0, y0 = float(ca[:, 0].min()), float(ca[:, 1].min())
            x1, y1 = float(ca[:, 0].max()), float(ca[:, 1].max())
            bw, bh = x1 - x0, y1 - y0
            if bw < 30 or bh < 30 or bw > max_w * W or bh > max_h * H:
                continue
            cand.append({
                "cls": cls, "name": "BUILDING" if cls == 1 else "TREE",
                "box": [x0, y0, bw, bh], "area": n,
                "conf": float(pr.mean()), "n_points": n,
                "contour": contour, "color": clip.palette["colors"][cls],
            })
        cand.sort(key=lambda d: d["area"], reverse=True)
        # keep boxes that are not near-duplicates of already selected ones
        picked = []
        for d in cand:
            if len(picked) >= top:
                break
            if all(_box_iou(d["box"], q["box"]) < 0.45 for q in picked):
                picked.append(d)
        dets.extend(picked)

    # small objects: connected components
    for cls, min_area, top in ((9, 8, 2), (10, 6, 1)):
        m = (clip.cls == cls) & good
        if not m.any():
            continue
        H2, W2 = int(H * 0.5), int(W * 0.5)
        xs = np.clip((sx[m] * 0.5).astype(np.int32), 0, W2 - 1)
        ys = np.clip((sy[m] * 0.5).astype(np.int32), 0, H2 - 1)
        mask = np.zeros((H2, W2), np.uint8); mask[ys, xs] = 255
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        comps, labels = _component_contours(mask, 0.5, min_area, top)
        pr = clip.prob[m]
        comp_ids = labels[ys, xs]
        sums = np.bincount(comp_ids, weights=pr, minlength=labels.max() + 1)
        cnts = np.bincount(comp_ids, minlength=labels.max() + 1)
        for d in comps:
            i = d["mask_index"]
            x0, y0, bw, bh = d["box"]
            if bw < 8 or bh < 8 or bw > 0.2 * W or bh > 0.3 * H:
                continue
            dets.append({
                "cls": cls, "name": "VEHICLE" if cls == 9 else "PERSON",
                "box": [x0, y0, bw, bh], "area": float(cnts[i]),
                "conf": float(sums[i] / max(1, cnts[i])), "n_points": int(cnts[i]),
                "contour": d["contour"], "color": clip.palette["colors"][cls],
            })
    # order: buildings/trees by area, then small objects
    dets.sort(key=lambda d: ({1: 0, 5: 1, 9: 2, 10: 3}[d["cls"]], -d["area"]))
    return dets[:8]


def _box_iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x0 = max(ax, bx); y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw); y1 = min(ay + ah, by + bh)
    iw = max(0.0, x1 - x0); ih = max(0.0, y1 - y0)
    inter = iw * ih
    return inter / max(1.0, aw * ah + bw * bh - inter)
