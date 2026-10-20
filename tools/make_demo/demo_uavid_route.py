#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""【原始离线批量渲染脚本副本 / legacy】

这个文件是从 ``SegStreamVGGT/demo/demo_uavid_route.py`` 复制过来的原始脚本，
保留用于「离线批量渲染 mp4 / 与网页版对照」。交互式网页版本请用::

    cd /data/luhangze/aic/make_demo
    python server.py --port 8787

网页版把推理（core.py / recon.py）与渲染（Three.js 前端）拆开了：
同一份点云/相机/语义缓存可以被浏览器实时旋转、切换视角、过滤类别和导出，
不需要为每个视角重新跑模型或 matplotlib。

注意：本文件仍依赖 matplotlib / scipy / open3d(可选)，且默认输出到
SegStreamVGGT/demo/output；日常使用建议直接用网页版。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

THIS = Path(__file__).resolve()
ROOT = Path(os.environ.get("SEGSTREAM_ROOT", "/data/luhangze/aic/SegStreamVGGT")).resolve()
sys.path.insert(0, str(ROOT))

from segstreamvggt.models.segstreamvggt import SegStreamVGGT          # noqa: E402
from segstreamvggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
from train_aic import load_checkpoint_into                          # noqa: E402

# AIC-11 调色板 (与 eval_aic.py 一致)
PALETTE = {
    0: (0, 0, 0), 1: (128, 0, 0), 2: (128, 64, 128), 3: (192, 192, 192),
    4: (128, 128, 64), 5: (0, 128, 0), 6: (128, 128, 0), 7: (192, 128, 64),
    8: (0, 0, 192), 9: (192, 0, 192), 10: (64, 64, 0), 11: (0, 192, 255),
}
NAMES = {0: "unlabeled", 1: "building", 2: "road", 3: "sidewalk", 4: "parking",
         5: "tree", 6: "low_veg", 7: "terrain", 8: "water", 9: "vehicle",
         10: "person", 11: "sky"}
BG = "#0b0f14"


def parse_hw(s: str):
    if "x" in str(s).lower():
        h, w = str(s).lower().split("x")
        return int(h), int(w)
    return int(s), int(s)


def build_model(ckpt: str, backbone: str, out_ratio: int, device: str, point_source: str = "depth"):
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
    # 1) StreamVGGT: aggregator + camera/depth(or point) heads
    load_checkpoint_into(model, backbone, "StreamVGGT backbone")
    # 2) SegStreamVGGT: query + semantic head (geometry 保持 1) 的权重)
    load_checkpoint_into(model, ckpt, "SegStreamVGGT seg")
    model.to(device).eval()
    return model


def load_frames(cache_root: str, dataset: str, split: str, seq: str,
                frame_ids, H: int, W: int, image_dir: str = "", max_frames: int = 0):
    names, rgbs = [], []
    if image_dir:
        paths = sorted(Path(image_dir).glob("*.png")) + sorted(Path(image_dir).glob("*.jpg"))
        if max_frames and len(paths) > max_frames:
            step = len(paths) / max_frames
            paths = [paths[min(len(paths) - 1, int(i * step))] for i in range(max_frames)]
        for p in paths:
            im = Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR)
            rgbs.append(np.asarray(im))
            names.append(p.stem)
    else:
        for fid in frame_ids:
            name = f"{seq}_{int(fid):06d}"
            p = os.path.join(cache_root, dataset, split, "img", name + ".jpg")
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            im = Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR)
            rgbs.append(np.asarray(im))
            names.append(name)
    x = torch.from_numpy(np.stack(rgbs).astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    return x.unsqueeze(0), rgbs, names


@torch.no_grad()
def infer(model, images, H: int, W: int, point_source: str = "depth"):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(images, with_geometry=True)
    seg = out["seg_prob"][0].float()                             # C,S,h,w
    C, S, h, w = seg.shape
    # 逐帧上采样到 H,W -> argmax + max prob
    seg_up = F.interpolate(seg.permute(1, 0, 2, 3), size=(H, W), mode="bilinear",
                           align_corners=False)                  # S,C,H,W
    prob, pred = seg_up.max(dim=1)                               # S,H,W
    pred = pred.cpu().numpy(); prob = prob.cpu().numpy()

    pose = out["pose_enc"].float()
    ext, intr = pose_encoding_to_extri_intri(pose, image_size_hw=(H, W))
    R = ext[0, :, :3, :3].cpu().numpy()
    t = ext[0, :, :3, 3].cpu().numpy()
    cams = -np.einsum("sji,sj->si", R, t)                        # S,3
    fov = 2.0 * np.degrees(np.arctan(H / (2.0 * intr[0, :, 1, 1].cpu().numpy())))

    if point_source == "depth":
        # depth + pose 反投影: 与相机同尺度, 点云是连续街道 (point head 的 world_points
        # 是另一套归一化尺度, 直接叠加会出现重影/尺度错位)
        dep = out["depth"][0, :, :, :, 0].float().cpu().numpy()          # S,H,W
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


def extract_points(wp, cf, pred, prob, rgbs, conf_q: float, seg_thresh: float,
                   drop_sky: bool, max_points: int):
    """返回 per-frame dict(points, rgb, cls, seg_rgb) 及全局 bounds。"""
    S, H, W, _ = wp.shape
    frames = []
    total = 0
    for s in range(S):
        m = np.isfinite(wp[s]).all(-1) & (cf[s] > np.quantile(cf[s], conf_q))
        m &= (prob[s] >= seg_thresh) & (pred[s] > 0)
        if drop_sky:
            m &= pred[s] != 11
        p = wp[s][m]
        cls = pred[s][m]
        rgb = rgbs[s][m]
        seg_rgb = np.array([PALETTE[int(c)] for c in cls], dtype=np.uint8)
        frames.append({"points": p, "rgb": rgb, "cls": cls, "seg_rgb": seg_rgb})
        total += len(p)
    # 全局随机下采样 (固定种子, 便于复现)
    rng = np.random.default_rng(0)
    if total > max_points:
        keep = rng.choice(total, size=max_points, replace=False)
        keep = np.sort(keep)
        # 按帧重新分配
        off = 0
        for f in frames:
            n = len(f["points"])
            k = keep[(keep >= off) & (keep < off + n)] - off
            for key in ("points", "rgb", "cls", "seg_rgb"):
                f[key] = f[key][k]
            off += n
    allp = np.concatenate([f["points"] for f in frames], 0)
    # VGGT world 坐标是 y-down; 显示时用 (x, z, -y) 让地面朝上
    disp = np.stack([allp[:, 0], allp[:, 2], -allp[:, 1]], axis=1)
    lo = np.percentile(disp, 1, axis=0)
    hi = np.percentile(disp, 99, axis=0)
    pad = (hi - lo) * 0.05 + 1e-6
    bounds = (lo - pad, hi + pad)
    return frames, bounds


def style_3d(ax, bounds, elev, azim, proj="persp"):
    ax.set_facecolor(BG)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho" if proj == "ortho" else "persp")
    (lo, hi) = bounds
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))


def smooth_route(cams, n=500):
    """把相机中心拟合成按弧长参数化的平滑曲线 (PCHIP, 不过冲)。"""
    from scipy.interpolate import PchipInterpolator
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(cams, axis=0), axis=1))]
    if d[-1] < 1e-9:
        return cams.copy(), d, None
    t = np.linspace(0.0, float(d[-1]), n)
    spl = PchipInterpolator(d, cams, axis=0)
    return spl(t), t, spl


def _disp(v):
    """VGGT world (x,y,z) -> 显示坐标 (x,z,-y)。"""
    return np.array([v[0], v[2], -v[1]])


def render_frame(k, u, frames, cams_disp, route, route_t, R, rgbs, names, bounds, out_png, args,
                 u_step=1.0):
    """渲染第 k→k+1 个航点之间、进度 u∈[0,1) 的一帧。"""
    K = len(frames)
    base = frames[:k + 1]
    pts = np.concatenate([f["points"] for f in base], 0)
    col_rgb = np.concatenate([f["rgb"] for f in base], 0).astype(np.float32) / 255.0
    col_seg = np.concatenate([f["seg_rgb"] for f in base], 0).astype(np.float32) / 255.0

    # 平滑补全: 下一航点的点按“离该航点相机由近到远”逐批加入。
    new_mask = np.zeros(len(pts), dtype=bool)
    if k < K - 1 and u > 0:
        n_base = len(pts)
        nxt = frames[k + 1]
        q = nxt["points"]
        q_disp = np.stack([q[:, 0], q[:, 2], -q[:, 1]], axis=1)
        # 相对“当前相机”的距离阈值: 只把超过阈值的新区域逐批加入,
        # 已经覆盖的近处不再重复堆叠。
        d = np.linalg.norm(q_disp - cams_disp[k], axis=1)
        keep = d >= float(args.min_new_dist)
        if keep.sum() == 0:
            keep = np.ones_like(d, dtype=bool)
        order = np.argsort(d[keep])
        order = np.where(keep)[0][order]
        n_add = int(round(u * len(order)))
        n_prev = int(round(max(0.0, u - u_step) * len(order)))
        take = order[:n_add]
        pts = np.concatenate([pts, q[take]], 0)
        col_rgb = np.concatenate([col_rgb, nxt["rgb"][take].astype(np.float32) / 255.0], 0)
        col_seg = np.concatenate([col_seg, nxt["seg_rgb"][take].astype(np.float32) / 255.0], 0)
        new_mask = np.zeros(len(pts), dtype=bool)
        new_mask[n_base + n_prev:] = True

    pts = np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1)
    if len(pts) > args.max_render:
        rng = np.random.default_rng(k * 1000 + int(u * 100))
        j = rng.choice(len(pts), args.max_render, replace=False)
        pts, col_rgb, col_seg, new_mask = pts[j], col_rgb[j], col_seg[j], new_mask[j]
    col_rgb = np.clip((col_rgb ** 0.62) * 1.15, 0.0, 1.0)
    if args.highlight_new and new_mask.any():
        col_rgb[new_mask] = np.clip(0.25 * col_rgb[new_mask] + 0.75 * np.array([1.0, 0.86, 0.20]), 0, 1)
        col_seg[new_mask] = np.array([1.0, 0.92, 0.25], dtype=np.float32)
    point_sizes = np.where(new_mask, 3.4, 0.9)

    # 平滑航迹
    s_cur = float((k + u) / max(K - 1, 1)) * float(route_t[-1])
    ri = int(np.searchsorted(route_t, s_cur, side="right") - 1)
    ri = max(0, min(ri, len(route) - 1))
    cur_cam = route[ri]
    route_travel = route[route_t <= s_cur + 1e-9]
    if len(route_travel) == 0:
        route_travel = route[:1]
    azim = args.view_azim + args.azim_drift * (k + u)
    if args.follow_route:
        j = int(np.argmin(np.abs(route_t - s_cur)))
        j0, j1 = max(0, j - 1), min(len(route) - 1, j + 1)
        tan = route[j1] - route[j0]
        azim = float(np.degrees(np.arctan2(tan[1], tan[0])))

    fig = plt.figure(figsize=(15, 7.4), facecolor=BG)
    ax1 = fig.add_axes([0.02, 0.075, 0.46, 0.79], projection="3d")
    ax2 = fig.add_axes([0.51, 0.075, 0.46, 0.79], projection="3d")

    # 相机视锥: 当前相机朝向的 4 条射线
    fwd = _disp(R[k].T @ np.array([0.0, 0.0, 1.0]))
    right = _disp(R[k].T @ np.array([1.0, 0.0, 0.0]))
    up = _disp(R[k].T @ np.array([0.0, 1.0, 0.0]))
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    right = right / (np.linalg.norm(right) + 1e-9)
    up = up / (np.linalg.norm(up) + 1e-9)

    for ax, col, title in (
        (ax1, col_rgb, "RGB point cloud"),
        (ax2, col_seg, "Semantic point cloud"),
    ):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=col, s=point_sizes, marker=".",
                   linewidths=0, alpha=0.95, depthshade=False)
        style_3d(ax, bounds, args.view_elev, azim, args.proj)
        # 全航迹(灰) + 已飞(红) + 当前无人机(黄)
        ax.plot(route[:, 0], route[:, 1], route[:, 2], color="#6b7280",
                lw=1.3, alpha=0.75)
        ax.plot(route_travel[:, 0], route_travel[:, 1], route_travel[:, 2],
                color="#ff4d4d", lw=2.6, alpha=0.98)
        ax.scatter([cur_cam[0]], [cur_cam[1]], [cur_cam[2]], c="#ffe066", s=75,
                   edgecolors="white", linewidths=0.8, depthshade=False)
        for sx in (-0.35, 0.35):
            for sy in (-0.35, 0.35):
                tip = cur_cam + 0.10 * (fwd + sx * right + sy * up)
                ax.plot([cur_cam[0], tip[0]], [cur_cam[1], tip[1]],
                        [cur_cam[2], tip[2]], color="#67e8f9", lw=1.1, alpha=0.95)

    # 当前航点的无人机视角 (左上)
    idx_show = min(k + (1 if u >= 0.5 else 0), K - 1)
    axi = fig.add_axes([0.015, 0.66, 0.155, 0.175])
    axi.imshow(rgbs[idx_show]); axi.set_xticks([]); axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_color("#ffffff"); sp.set_linewidth(1.2)
    axi.set_title(f"UAV view  {names[idx_show]}", color="white", fontsize=7.0, pad=2)

    # 航线小地图 (左下): 全航迹 / 已飞 / 当前
    axm = fig.add_axes([0.018, 0.06, 0.155, 0.165])
    axm.set_facecolor("#0f1620")
    axm.plot(route[:, 0], route[:, 1], color="#6b7280", lw=1.2)
    axm.plot(route_travel[:, 0], route_travel[:, 1], color="#ff4d4d", lw=2.4)
    axm.scatter([cur_cam[0]], [cur_cam[1]], c="#ffe066", s=34,
                edgecolors="white", linewidths=0.6, zorder=5)
    axm.set_aspect("equal"); axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_color("#334155"); sp.set_linewidth(0.8)
    axm.set_title("Flight route", color="#dfe6ee", fontsize=7.0, pad=2)

    pct = int(round((k + u) / max(K - 1, 1) * 100))
    fig.text(0.5, 0.965, f"{args.title} {args.seq_label} — incremental 3D reconstruction  "
             f"waypoint {k + 1}/{K}  ({pct}%)", color="white", ha="center",
             fontsize=14, weight="bold")
    fig.text(0.5, 0.935, "frozen StreamVGGT geometry + trained SegStreamVGGT query head  |  "
             f"{args.model_label}  |  "
             f"{('depth+pose unprojection' if args.point_source == 'depth' else 'point-head world_points')}"
             "  |  smooth route (PCHIP) + near-to-far reveal",
             color="#aab6c6", ha="center", fontsize=8.2)
    fig.text(0.25, 0.895, f"RGB point cloud  |  {len(pts):,} pts", color="#dfe6ee",
             ha="center", fontsize=10, weight="bold")
    fig.text(0.75, 0.895, f"Semantic point cloud  |  {len(pts):,} pts", color="#dfe6ee",
             ha="center", fontsize=10, weight="bold")
    handles = [Patch(facecolor=np.array(PALETTE[c]) / 255.0, label=NAMES[c])
               for c in range(1, 12)]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.005),
               frameon=False, fontsize=7.5, labelcolor="#dfe6ee")
    fig.savefig(out_png, dpi=args.dpi, facecolor=BG)
    plt.close(fig)


def write_ply(path, pts, colors):
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
    except Exception as e:  # noqa: BLE001
        import trimesh
        trimesh.PointCloud(pts, colors=colors).export(str(path))
        print(f"[PLY] open3d 失败({e}), 用 trimesh 写入", flush=True)


def main():
    ap = argparse.ArgumentParser(description="UAV 航迹增量 RGB/语义 3D 点云 demo")
    ap.add_argument("--ckpt", default=str(ROOT / "logs_aic_v5/checkpoint_best.pth"))
    ap.add_argument("--backbone", default="/data/luhangze/aic/StreamVGGT/ckpt/checkpoints.pth")
    ap.add_argument("--model-label", default="v5")
    ap.add_argument("--cache-root", default=str(ROOT / "cache"))
    ap.add_argument("--dataset", default="uavid")
    ap.add_argument("--split", default="val")
    ap.add_argument("--seq", default="seq16")
    ap.add_argument("--seq-label", default="seq16")
    ap.add_argument("--title", default="UAVid")
    ap.add_argument("--frames", default="0,100,200,300,400,500,600,700,800,900")
    ap.add_argument("--image-dir", default="",
                    help="可选: 直接读取该目录下的 png/jpg (例如从 images.mp4 抽的连续帧); 此时忽略 --frames")
    ap.add_argument("--max-frames", type=int, default=0, help="配合 --image-dir, 最多用多少帧 (0=全部)")
    ap.add_argument("--img-size", default="392x700", help="HxW, 必须是 14 的倍数")
    ap.add_argument("--out", default=str(ROOT / "demo/output"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-ratio", type=int, default=2)
    ap.add_argument("--conf-quantile", type=float, default=0.10,
                    help="每帧保留 confidence 高于该分位数的点")
    ap.add_argument("--seg-thresh", type=float, default=0.0,
                    help="语义 argmax 概率低于该值的点丢弃")
    ap.add_argument("--max-points", type=int, default=1500000, help="全局最多保留点数")
    ap.add_argument("--max-render", type=int, default=400000, help="每帧渲染最多点数")
    ap.add_argument("--view-elev", type=float, default=15)
    ap.add_argument("--view-azim", type=float, default=-55)
    ap.add_argument("--azim-drift", type=float, default=2.2,
                    help="视角方位每航点额外旋转的角度 (0=固定视角, 不晃动)")
    ap.add_argument("--follow-route", type=int, default=1,
                    help="视角方位跟随飞行方向 (更像无人机沿航线看)")
    ap.add_argument("--proj", default="persp", choices=["persp", "ortho"],
                    help="斜向下看城市用 persp; 纯俯视地图用 ortho")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--highlight-new", type=int, default=1,
                    help="把当前子帧新增的点高亮成黄色, 便于看清新区域")
    ap.add_argument("--min-new-dist", type=float, default=0.15,
                    help="只新增距当前相机超过该阈值的新点 (世界显示尺度, 场景约 1-3)")
    ap.add_argument("--subframes", type=int, default=5,
                    help="每个航点之间插值的子帧数 (越大越顺滑, 渲染越慢)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--drop-sky", type=int, default=1)
    ap.add_argument("--export-npz", default="", help="把逐帧点云/相机位姿导出为 npz")
    ap.add_argument("--export-only", action="store_true", help="只导出 npz, 不渲染")
    ap.add_argument("--point-source", default="depth", choices=["depth", "point"],
                    help="depth=depth+pose 反投影 (推荐, 街道连续); point=point head world_points")
    args = ap.parse_args()

    H, W = parse_hw(args.img_size)
    assert H % 14 == 0 and W % 14 == 0, "--img-size 必须是 14 的倍数"
    out = Path(args.out); (out / "frames").mkdir(parents=True, exist_ok=True)
    frame_ids = [int(x) for x in str(args.frames).split(",") if x.strip()]

    print(f"[DEMO] ckpt={args.ckpt}\n[DEMO] seq={args.seq_label} frames={len(frame_ids) if not args.image_dir else args.image_dir} input={H}x{W}", flush=True)
    model = build_model(args.ckpt, args.backbone, args.out_ratio, args.device, args.point_source)
    images, rgbs, names = load_frames(args.cache_root, args.dataset, args.split,
                                      args.seq, frame_ids, H, W, args.image_dir, args.max_frames)
    images = images.to(args.device)
    wp, cf, pred, prob, cams, R, fov = infer(model, images, H, W, args.point_source)
    print(f"[DEMO] world_points={wp.shape} conf=[{cf.min():.3f},{cf.max():.3f}] "
          f"cam_centers={len(cams)}", flush=True)
    frames, bounds = extract_points(wp, cf, pred, prob, rgbs, args.conf_quantile,
                                    args.seg_thresh, bool(args.drop_sky), args.max_points)
    print("[DEMO] points per waypoint:", [len(f["points"]) for f in frames], flush=True)

    if args.export_npz:
        offs = np.cumsum([0] + [len(f["points"]) for f in frames]).astype(np.int64)
        np.savez_compressed(
            args.export_npz,
            points=np.concatenate([f["points"] for f in frames], 0).astype(np.float32),
            rgb=np.concatenate([f["rgb"] for f in frames], 0).astype(np.uint8),
            seg=np.concatenate([f["seg_rgb"] for f in frames], 0).astype(np.uint8),
            offsets=offs, cams=cams.astype(np.float32), R=R.astype(np.float32),
            fov=fov.astype(np.float32), names=np.array(names), H=H, W=W,
        )
        print(f"[DEMO] export -> {args.export_npz}", flush=True)
        if args.export_only:
            return

    # 平滑航迹 (PCHIP, 按弧长) + 航点间子帧
    cams_disp = np.stack([cams[:, 0], cams[:, 2], -cams[:, 1]], axis=1)
    route, route_t, _ = smooth_route(cams_disp, n=500)
    sub = max(1, int(args.subframes))
    K = len(frames)
    n_anim = (K - 1) * sub + 1
    print(f"[DEMO] animation: {K} waypoints x {sub} subframes = {n_anim} frames", flush=True)
    for a in range(n_anim):
        k = min(a // sub, K - 1)
        u = (a % sub) / sub if k < K - 1 else 0.0
        render_frame(k, u, frames, cams_disp, route, route_t, R, rgbs, names, bounds,
                     out / "frames" / f"frame_{a:03d}.png", args, u_step=1.0 / sub)
        if (a + 1) % 10 == 0 or a == n_anim - 1:
            print(f"[DEMO] rendered {a + 1}/{n_anim}", flush=True)

    # 预览拼图: 4 个阶段
    idx = [0, n_anim // 3, 2 * n_anim // 3, n_anim - 1]
    ims = [Image.open(out / "frames" / f"frame_{i:03d}.png").convert("RGB") for i in idx]
    w, h = ims[0].size
    grid = Image.new("RGB", (w * 2, h * 2), (11, 15, 20))
    for k, im in enumerate(ims):
        grid.paste(im, ((k % 2) * w, (k // 2) * h))
    grid.save(out / "preview_grid.png")

    # 最终点云 PLY
    pts = np.concatenate([f["points"] for f in frames], 0)
    rgb = np.concatenate([f["rgb"] for f in frames], 0)
    seg = np.concatenate([f["seg_rgb"] for f in frames], 0)
    write_ply(out / "points_rgb.ply", pts, rgb)
    write_ply(out / "points_seg.ply", pts, seg)

    # 视频
    mp4 = out / "uavid_seq16_route.mp4"
    if not args.no_video:
        cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
               "-i", str(out / "frames" / "frame_%03d.png"),
               "-vf", "tpad=stop_mode=clone:stop_duration=2",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[DEMO] video -> {mp4}", flush=True)

    summary = {
        "ckpt": args.ckpt, "model_label": args.model_label, "seq": args.seq_label,
        "frames": frame_ids, "input_hw": [H, W],
        "points_per_waypoint": [len(f["points"]) for f in frames],
        "subframes": sub, "animation_frames": n_anim,
        "points_total": int(len(pts)),
        "conf_quantile": args.conf_quantile, "seg_thresh": args.seg_thresh,
        "point_source": args.point_source, "view_elev": args.view_elev,
        "view_azim": args.view_azim, "proj": args.proj, "follow_route": args.follow_route,
        "azim_drift": args.azim_drift,
        "min_new_dist": args.min_new_dist,
        "highlight_new": args.highlight_new,
        "drop_sky": bool(args.drop_sky),
        "camera_centers": cams.round(5).tolist(),
        "outputs": {"video": str(mp4), "points_rgb": str(out / "points_rgb.ply"),
                    "points_seg": str(out / "points_seg.ply"),
                    "preview": str(out / "preview_grid.png")},
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:1200])


if __name__ == "__main__":
    main()
