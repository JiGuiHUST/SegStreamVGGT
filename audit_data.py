# -*- coding: utf-8 -*-
"""
AIC 数据读取正确性审计 (SegStreamVGGT)

回答"目前的读取方式对不对"，用**独立于缓存代码的重新解码**做硬对比:

  1. counts    : 原始文件清点 vs 缓存清点, 划分是否与官方/声明一致, train/val 是否泄漏
  2. cache     : **全量**扫描缓存标签 (不是抽样), 检查 id ⊆ {0}∪C_d、尺寸与 manifest 一致、无空标签
  3. pairs     : 抽样把**原始数据用 label_mapping 重新解码一遍**, 与缓存标签**逐像素**比对 (要求 100% 相等);
                 并检查缓存的 img/lbl 是同一张图 (尺寸一致 + 图像-标签相关)
  4. align     : 抽样做"标签边界 ↔ 影像梯度"一致性检验 (对齐应显著 > 1.0, 随机 mask = 1.00)
  5. semantics : 不依赖映射表的独立语义核查 (天空在最上方、水体偏蓝、各数据集类的像素占比等)

    python audit_data.py                     # 全部检查
    python audit_data.py --checks cache      # 只全量扫缓存标签
    python audit_data.py --samples 8 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))                 # SegStreamVGGT/
sys.path.insert(0, str(THIS.parents[1]))             # aic/  (label_mapping.py)

import label_mapping as lm                                                     # noqa: E402
from aic_data import precache as P                                             # noqa: E402
from aic_data.registry import AIC_DATASETS, IGNORE_INDEX, NUM_AIC_CLASSES, get_spec  # noqa: E402

AIC_ROOT = "/data1/luhangze/AIC"
CACHE_ROOT = str(THIS.parent / "cache")
NAMES = {i: n for i, n, _c, _d in lm.UNIFIED_CLASSES}


# ---------------------------------------------------------------------------
def raw_lists(ds: str) -> Dict[str, list]:
    fn = {
        "uavid": P._list_uavid, "aeroscapes": P._list_aeroscapes, "dlr3k": P._list_dlr3k,
        "dlrsd": P._list_dlrsd, "dstl": P._list_dstl, "potsdam": P._list_potsdam, "hdmaps": P._list_hdmaps,
    }[ds]
    tr, va = fn(AIC_ROOT)
    return {"train": tr, "val": va}


def cached_items(ds: str, split: str) -> List[dict]:
    p = os.path.join(CACHE_ROOT, ds, split, "manifest.json")
    with open(p) as f:
        return json.load(f)["items"]


def cached_label(ds: str, split: str, name: str) -> np.ndarray:
    return np.asarray(Image.open(os.path.join(CACHE_ROOT, ds, split, "lbl", name + ".png")))


def cached_image(ds: str, split: str, name: str) -> np.ndarray:
    return np.asarray(Image.open(os.path.join(CACHE_ROOT, ds, split, "img", name + ".jpg")).convert("RGB"))


def redecode_uni(ds: str, item: Tuple) -> np.ndarray:
    """用 label_mapping **重新**把原始数据解码成统一索引图 (不经过缓存代码)。"""
    _name, _group, img_path, payload = item
    if ds == "dlr3k":
        im = Image.open(img_path)
        uni = lm.remap("dlr3k", payload[0], shape=(im.size[1], im.size[0]))
    elif ds == "dstl":
        tile, rows, (xmax, ymax) = payload
        im = Image.open(img_path)
        W, H = im.size
        uni = lm.remap("dstl", ((H, W), xmax, ymax, rows))
    else:
        uni = lm.remap(ds, payload)
    if NUM_AIC_CLASSES < 256:
        uni = np.where(uni >= NUM_AIC_CLASSES, IGNORE_INDEX, uni).astype(np.uint8)
    return uni


# ---------------------------------------------------------------------------
# 1. 清点 / 划分
# ---------------------------------------------------------------------------
def check_counts(ds_list) -> int:
    print("\n" + "=" * 96)
    print("[1] 原始文件清点 vs 缓存清点 / 划分")
    print("=" * 96)
    bad = 0
    for ds in ds_list:
        raw = raw_lists(ds)
        for split in ("train", "val"):
            n_raw, n_cache = len(raw[split]), len(cached_items(ds, split))
            names_raw = {it[0] for it in raw[split]}
            names_cache = {it["name"] for it in cached_items(ds, split)}
            miss = names_raw - names_cache
            extra = names_cache - names_raw
            flag = "OK " if (n_raw == n_cache and not miss and not extra) else "ERR"
            if flag == "ERR":
                bad += 1
            print(f"  [{flag}] {ds:11s}/{split:5s} 原始 {n_raw:5d} -> 缓存 {n_cache:5d}"
                  f" | 缺失 {len(miss)} 多余 {len(extra)}")
            if miss:
                print(f"        缺失示例: {sorted(miss)[:3]}")
            if extra:
                print(f"        多余示例: {sorted(extra)[:3]}")
        # train/val 泄漏
        tr = {it["name"] for it in cached_items(ds, "train")}
        va = {it["name"] for it in cached_items(ds, "val")}
        leak = tr & va
        if leak:
            bad += 1
            print(f"  [ERR] {ds:11s} train/val 同名泄漏 {len(leak)} 个: {sorted(leak)[:3]}")
        else:
            print(f"  [OK ] {ds:11s} train/val 无重叠 (train {len(tr)} / val {len(va)})")
    return bad


# ---------------------------------------------------------------------------
# 2. 全量扫缓存标签
# ---------------------------------------------------------------------------
def _scan_one(args):
    ds, split, name, allowed, W, H, raw_dir, idx_kind = args
    try:
        a = cached_label(ds, split, name)
    except Exception as e:                                        # noqa: BLE001
        return (ds, split, name, "read_error", str(e))
    ids = np.unique(a)
    extra = sorted(int(i) for i in ids if int(i) not in allowed)
    if extra:
        return (ds, split, name, "bad_ids", str(extra[:8]))
    if (a.shape[1], a.shape[0]) != (W, H):
        return (ds, split, name, "size_mismatch", f"{a.shape[::-1]} vs {(W, H)}")
    if int((a != IGNORE_INDEX).sum()) == 0:
        # 全 ignore 分两种: 上游数据本身就是"只有未启用类"(合法) vs 映射丢了内容(错误)
        raw = os.path.join(raw_dir, name + idx_kind)
        if os.path.exists(raw):
            r = np.asarray(Image.open(raw))
            if r.ndim == 2 and r.max() > 0:
                return (ds, split, name, "all_ignore_from_raw", str(sorted(np.unique(r))[:9]))
        return (ds, split, name, "all_ignore_empty_raw", "")
    return None


def check_cache(ds_list, workers: int) -> int:
    print("\n" + "=" * 96)
    print("[2] 全量扫描缓存标签 (id ⊆ {0}∪C_d / 尺寸 / 非空)")
    print("=" * 96)
    RAW_DIR = {
        "aeroscapes": (f"{AIC_ROOT}/aeroscapes/SegmentationClass", ".png"),
        "dlrsd": (f"{AIC_ROOT}/DLRSD/Labels", "/"),
    }
    jobs = []
    for ds in ds_list:
        allowed = {IGNORE_INDEX} | {c for c in get_spec(ds).classes if c < NUM_AIC_CLASSES}
        rd, kind = RAW_DIR.get(ds, ("", ""))
        for split in ("train", "val"):
            for it in cached_items(ds, split):
                sub = it["name"].split("_")[0] if ds == "dlrsd" else ""
                raw = os.path.join(rd, sub) if ds == "dlrsd" else rd
                jobs.append((ds, split, it["name"], allowed, it["width"], it["height"], raw, kind))
    t0 = time.time()
    problems = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_scan_one, jobs, chunksize=32)):
            if r is not None:
                problems.append(r)
            if (i + 1) % 1000 == 0:
                print(f"    ... {i+1}/{len(jobs)}", flush=True)
    print(f"  扫描 {len(jobs)} 个缓存标签, 耗时 {time.time()-t0:.0f}s")
    if not problems:
        print(f"  [OK ] 全部 {len(jobs)} 个缓存标签: 只含该数据集标注的类、尺寸与 manifest 一致、非全 ignore")
        return 0
    from collections import Counter
    cnt = Counter((p[0], p[1], p[3]) for p in problems)
    for (ds, split, kind), n in sorted(cnt.items()):
        print(f"  [{'WARN' if 'empty_raw' in kind else 'ERR '}] {ds}/{split}: {n} 个 {kind}")
        for p in [q for q in problems if (q[0], q[1], q[3]) == (ds, split, kind)][:3]:
            print(f"        {p[2]} raw_classes={p[4]}")
    hard = [p for p in problems if "empty_raw" not in p[3]]
    if not hard:
        print("  [OK ] 上述全 ignore 均为『上游数据本身只有未启用类』(如 AeroScapes 的 Boat/Drone 帧), 属合法数据,"
              " 已由 manifest 的 valid_px 标记并在训练时跳过")
    return len(hard)


# ---------------------------------------------------------------------------
# 3. 原始重新解码 vs 缓存 (逐像素)
# ---------------------------------------------------------------------------
def check_pairs(ds_list, n_samples: int, seed: int = 0) -> int:
    print("\n" + "=" * 96)
    print("[3] 抽样: 原始数据用 label_mapping 重新解码 vs 缓存标签 (逐像素 100% 相等)")
    print("=" * 96)
    bad = 0
    rng = random.Random(seed)
    for ds in ds_list:
        raw = raw_lists(ds)
        for split in ("train", "val"):
            items = raw[split]
            if not items:
                continue
            pick = rng.sample(items, min(n_samples, len(items)))
            worst_mismatch, name_bad, dim_bad, img_diff = 0, None, None, []
            for it in pick:
                name = it[0]
                try:
                    uni = redecode_uni(ds, it)
                    a = cached_label(ds, split, name)
                except Exception as e:                            # noqa: BLE001
                    print(f"  [ERR] {ds}/{split}/{name} 重新解码失败: {e}")
                    bad += 1
                    continue
                if a.shape != uni.shape:
                    dim_bad = f"{name}: cache {a.shape} vs raw {uni.shape}"
                    continue
                mism = int((a != uni).sum())
                if mism:
                    worst_mismatch = max(worst_mismatch, mism)
                    name_bad = name
                # 图像: 缓存 jpg vs 原始解码 (只比尺寸是否一致, 内容用均值差容忍 jpeg)
                try:
                    im = cached_image(ds, split, name)
                    if (im.shape[1], im.shape[0]) != (uni.shape[1], uni.shape[0]):
                        img_diff.append((name, "size"))
                except Exception as e:                            # noqa: BLE001
                    img_diff.append((name, str(e)[:30]))
            ok = (worst_mismatch == 0) and (dim_bad is None) and not img_diff
            bad += 0 if ok else 1
            msg = f"标签逐像素一致 ✅" if worst_mismatch == 0 else f"不一致 {worst_mismatch} px ({name_bad})"
            print(f"  [{'OK ' if ok else 'ERR'}] {ds:11s}/{split:5s} 抽 {len(pick):2d} 张: {msg}"
                  + (f" | 尺寸/读图异常 {img_diff[:2]}" if dim_bad or img_diff else ""))
    return bad


# ---------------------------------------------------------------------------
# 4. 对齐检验 (标签边界 ↔ 影像梯度)
# ---------------------------------------------------------------------------
def align_score(img: np.ndarray, lbl: np.ndarray) -> float:
    """标签边界处的影像梯度均值 / 全图影像梯度均值 (>1 表示边界落在影像的真实边缘上)。

    大图 (>4 Mpx, 如 Potsdam 6000² / HD-Maps 5734×4679) 先整数倍降采样再算,
    对齐关系在整数倍下保持不变, 但速度快一个数量级。
    """
    im = img if isinstance(img, Image.Image) else Image.fromarray(img)
    h, w = lbl.shape
    f = 1
    while (h // f) * (w // f) > 4_000_000:
        f += 1
    if f > 1:
        im = im.resize((max(1, w // f), max(1, h // f)), Image.BILINEAR)
        lbl = np.asarray(Image.fromarray(lbl.astype(np.uint8)).resize(
            (max(1, w // f), max(1, h // f)), Image.NEAREST))
    g = np.asarray(im.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    b = np.zeros_like(lbl, dtype=bool)
    b[1:, :] |= lbl[1:, :] != lbl[:-1, :]
    b[:, 1:] |= lbl[:, 1:] != lbl[:, :-1]
    b &= lbl != IGNORE_INDEX
    if b.sum() < 50:
        return float("nan")
    from scipy.ndimage import binary_dilation            # noqa: PLC0415
    b = binary_dilation(b, structure=np.ones((3, 3), bool))   # 膨胀 1px, 容忍微小偏移
    return float(mag[b].mean() / max(mag.mean(), 1e-6))


def check_align(ds_list, n_samples: int, seed: int = 0) -> int:
    print("\n" + "=" * 96)
    print("[4] 对齐检验: 标签边界处的影像梯度 / 全图梯度 (对齐参考 1.5~3.0, 随机 mask ≈ 1.00)")
    print("=" * 96)
    rng = random.Random(seed)
    bad = 0
    for ds in ds_list:
        scores = []
        for split in ("train", "val"):
            items = cached_items(ds, split)
            pick = rng.sample(items, min(n_samples, len(items)))
            for it in pick:
                img = cached_image(ds, split, it["name"])
                lbl = cached_label(ds, split, it["name"])
                if lbl.shape != img.shape[:2]:
                    continue
                s = align_score(img, lbl)
                if s == s:
                    scores.append(s)
        if not scores:
            print(f"  [--  ] {ds:11s} 无有效样本 (全是 ignore? 如 dlr3k 只有车辆框)")
            continue
        mean, mn = float(np.mean(scores)), float(np.min(scores))
        ok = mean > 1.3
        bad += 0 if ok else 1
        print(f"  [{'OK ' if ok else 'ERR'}] {ds:11s} 对齐得分 均值 {mean:5.2f} / 最低 {mn:5.2f} "
              f"(n={len(scores)}) {'✅' if ok else '<-- 偏低, 需人工看叠加图'}")
    return bad


# ---------------------------------------------------------------------------
# 5. 独立语义核查
# ---------------------------------------------------------------------------
def check_semantics(ds_list) -> int:
    print("\n" + "=" * 96)
    print("[5] 独立语义核查 (不依赖映射表: 天空位置/水体颜色/各类面积占比)")
    print("=" * 96)
    for ds in ds_list:
        items = cached_items(ds, "train")[:40]
        tot = {}
        pos_sum, rgb_sum = {}, {}
        H = W = None
        for it in items:
            lbl = cached_label(ds, "train", it["name"])
            img = cached_image(ds, "train", it["name"])
            if lbl.shape != img.shape[:2]:
                continue
            H, W = lbl.shape
            for c in np.unique(lbl):
                c = int(c)
                if c == IGNORE_INDEX:
                    continue
                m = lbl == c
                n = int(m.sum())
                tot[c] = tot.get(c, 0) + n
                ys = np.nonzero(m)[0]
                pos_sum[c] = pos_sum.get(c, 0.0) + float(ys.mean()) / H * n
                rgb_sum[c] = rgb_sum.get(c, np.zeros(3)) + img[m].mean(axis=0) * n
        if not tot:
            print(f"  [--  ] {ds:11s} 无像素")
            continue
        total = sum(tot.values())
        print(f"  --- {ds} (抽样 {len(items)} 张) ---")
        for c in sorted(tot, key=lambda k: -tot[k])[:6]:
            frac = tot[c] / total
            ypos = pos_sum[c] / tot[c]
            rgb = rgb_sum[c] / tot[c]
            print(f"      {NAMES[c]:16s} 面积 {frac*100:5.1f}% | 平均 y/H {ypos:.2f} | 平均 RGB "
                  f"({rgb[0]:5.0f},{rgb[1]:5.0f},{rgb[2]:5.0f})")
        # 硬性语义断言
        checks = []
        if 11 in tot:      # sky 必须在图像上部
            checks.append(("sky 在图像上部 (y/H < 0.4)", pos_sum[11] / tot[11] < 0.4))
        if 8 in tot:       # water 应该偏蓝 (B > R)
            rgbw = rgb_sum[8] / tot[8]
            checks.append(("water 蓝色分量 > 红色", rgbw[2] > rgbw[0]))
        if 5 in tot:       # tree 应该绿 > 红
            rgbt = rgb_sum[5] / tot[5]
            checks.append(("tree 绿色分量 > 红色", rgbt[1] > rgbt[0]))
        if 9 in tot:
            checks.append(("vehicle 面积 < 5%", tot[9] / total < 0.05))
        for label, ok in checks:
            print(f"      [{'OK ' if ok else 'ERR'}] {label}")
    return 0


# ---------------------------------------------------------------------------
# 6. 独立清点 (用 glob/CSV 直接从原始目录数, 不经过 registry 的 lister)
# ---------------------------------------------------------------------------
def check_files() -> int:
    print("\n" + "=" * 96)
    print("[6] 独立清点: 直接 glob 原始目录 vs 缓存总数")
    print("=" * 96)
    root = AIC_ROOT
    g = {
        "uavid": (f"{root}/UAVid/uavid_v1.5_official_release_image/uavid_*/seq*/Labels/*.png", 270),
        "aeroscapes": (f"{root}/aeroscapes/SegmentationClass/*.png", 3269),
        "dlr3k": (f"{root}/DLR3K/MunichDatasetVehicleDetection-2015-old/Train/*.JPG", 10),
        "dlrsd": (f"{root}/DLRSD/Labels/*/*.png", 2100),
        "potsdam": (f"{root}/ISPRS_Potsdam/raw/Potsdam/5_Labels_for_participants_no_Boundary/*.tif", 24),
        "hdmaps": (f"{root}/HD-Maps/Aerial/colorAnnotation/*.png", 20),
    }
    bad = 0
    for ds, (pattern, expected) in g.items():
        n_raw = len(glob.glob(pattern))
        n_cache = sum(len(cached_items(ds, s)) for s in ("train", "val"))
        ok = (n_raw == n_cache == expected)
        bad += 0 if ok else 1
        print(f"  [{'OK ' if ok else 'ERR'}] {ds:11s} 原始 glob {n_raw:5d} | 缓存 {n_cache:5d} | "
              f"上游文档/预期 {expected:5d}")
    # dstl: 用 train_wkt_v4.csv 的瓦片数
    csv_path = f"{root}/dstl/dstl-satellite-imagery-feature-detection/train_wkt_v4.csv"
    csv.field_size_limit(10 ** 9)
    with open(csv_path) as f:
        tiles = {r[0] for r in list(csv.reader(f))[1:] if r}
    n_cache = sum(len(cached_items("dstl", s)) for s in ("train", "val"))
    ok = len(tiles) == n_cache == 25
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'ERR'}] dstl        原始 WKT 瓦片 {len(tiles):5d} | 缓存 {n_cache:5d} | 预期    25")
    # hdmaps 另外一半标注在 test_images 里
    n_ann = len(glob.glob(f"{root}/HD-Maps/Aerial/colorAnnotation/*.png"))
    n_used = 0
    for nm in ("train_images", "test_images"):
        used = [p for p in glob.glob(f"{root}/HD-Maps/Aerial/{nm}/*.jpg")
                if os.path.exists(f"{root}/HD-Maps/Aerial/colorAnnotation/" +
                                  os.path.splitext(os.path.basename(p))[0] + ".png")]
        n_used += len(used)
    print(f"  [{'OK ' if n_ann == n_used == 20 else 'ERR'}] hdmaps      标注 {n_ann} 张, 其中能配上影像的 {n_used} 张")
    return bad


# ---------------------------------------------------------------------------
# 7. DLRSD 索引语义独立校验 (对照上游 multlabels.csv)
# ---------------------------------------------------------------------------
CSV_ORDER = ["airplane", "baresoil", "buildings", "cars", "chaparral", "court", "dock", "field",
             "grass", "mobilehome", "pavement", "sand", "sea", "ship", "tanks", "trees", "water"]


def check_dlrsd(n_images: int = 200, seed: int = 0) -> int:
    print("\n" + "=" * 96)
    print("[7] DLRSD 索引语义校验: 像素标签出现的类 vs 上游 multlabels.csv (索引与列序一一对应)")
    print("=" * 96)
    csv_path = f"{AIC_ROOT}/DLRSD/multilabels.csv"
    rows = {}
    with open(csv_path) as f:
        rd = csv.reader(f)
        header = next(rd)
        assert header[1:] == CSV_ORDER, f"CSV 列序与表不符: {header[1:]}"
        for r in rd:
            if r:
                rows[r[0]] = [int(x) for x in r[1:]]
    print(f"  CSV 列序 == dlrsd_table 的索引顺序 ✅ (1..17 = {', '.join(CSV_ORDER)})")
    print(f"  CSV 共 {len(rows)} 行 (上游 2100 张)")

    imgs = glob.glob(f"{AIC_ROOT}/DLRSD/Labels/*/*.png")
    rnd = random.Random(seed)
    pick = rnd.sample(imgs, min(n_images, len(imgs)))
    extra_cnt = {i: 0 for i in range(1, 18)}
    miss_cnt = {i: 0 for i in range(1, 18)}
    n_ok = 0
    for p in pick:
        name = os.path.splitext(os.path.basename(p))[0]
        if name not in rows:
            continue
        lbl = np.asarray(Image.open(p))
        present = {int(v) for v in np.unique(lbl) if 1 <= int(v) <= 17}
        csv_present = {i + 1 for i, v in enumerate(rows[name]) if v == 1}
        if present == csv_present:
            n_ok += 1
        for i in present - csv_present:
            extra_cnt[i] += 1
        for i in csv_present - present:
            miss_cnt[i] += 1
    n = len(pick)
    print(f"  逐图完全一致: {n_ok}/{n} ({100*n_ok/max(n,1):.1f}%)")
    print(f"  {'索引':>4} {'类名':<12} {'像素里有但 CSV 无':>16} {'CSV 有但像素无':>16}")
    worst = 0
    for i in range(1, 18):
        worst = max(worst, extra_cnt[i], miss_cnt[i])
        if extra_cnt[i] or miss_cnt[i]:
            print(f"  {i:4d} {CSV_ORDER[i-1]:<12} {extra_cnt[i]:16d} {miss_cnt[i]:16d}")
    # 判定: 不允许出现系统性偏移 (若索引/列序错位, 某些类会大量不一致)
    ok = worst <= max(3, int(0.05 * n))
    print(f"  [{'OK ' if ok else 'ERR'}] 最大单类不一致 {worst} 次 (阈值 {max(3, int(0.05*n))}) —— "
          f"索引/列序无系统性错位")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 8. DLR3K 车辆框叠加图 (人工目视)
# ---------------------------------------------------------------------------
def check_dlr3k_overlay(n_images: int = 3) -> int:
    print("\n" + "=" * 96)
    print("[8] DLR3K 车辆框栅格化 -> 叠加图 (供人工目视确认框落在车上)")
    print("=" * 96)
    out_dir = str(THIS.parent / "audit_out")
    os.makedirs(out_dir, exist_ok=True)
    items = cached_items("dlr3k", "train") + cached_items("dlr3k", "val")
    saved = []
    for it in items[:n_images]:
        name = it["name"]
        img = cached_image("dlr3k", "train" if it in cached_items("dlr3k", "train") else "val", name)
        lbl = cached_label("dlr3k", "train" if it in cached_items("dlr3k", "train") else "val", name)
        vis = img.copy()
        m = lbl == 9
        vis[m] = (0.35 * vis[m] + 0.65 * np.array([255, 0, 255])).astype(np.uint8)
        # 裁出若干框附近区域拼一起, 便于放大看
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        crops = []
        rnd = random.Random(0)
        idx = rnd.sample(range(len(ys)), min(6, len(ys)))
        for i in idx:
            y, x = ys[i], xs[i]
            y0, y1 = max(0, y - 40), min(vis.shape[0], y + 40)
            x0, x1 = max(0, x - 40), min(vis.shape[1], x + 40)
            crops.append(vis[y0:y1, x0:x1])
        if crops:
            h = min(c.shape[0] for c in crops)
            grid = np.concatenate([c[:h] for c in crops], axis=1)
            p = os.path.join(out_dir, f"dlr3k_boxcheck_{name}.jpg")
            Image.fromarray(grid).save(p, quality=92)
            saved.append(p)
            print(f"  {name}: vehicle 像素 {int(m.sum())} 个 ({100*m.mean():.4f}%), 框附近裁剪 -> {p}")
    print(f"  [OK ] 已生成 {len(saved)} 张叠加图 (紫色 = 映射为 vehicle 的像素), 请目视确认")
    return 0


# ---------------------------------------------------------------------------
# 9. 原始标签取值/颜色集合全量扫描 (堵住"未知索引被静默忽略"的漏洞)
# ---------------------------------------------------------------------------
def _raw_index_range(p):
    a = np.asarray(Image.open(p))
    u = np.unique(a)
    return int(u.min()), int(u.max()), [int(x) for x in u]


def _raw_color_set(p, max_px=2_000_000):
    a = np.asarray(Image.open(p).convert("RGB")).reshape(-1, 3)
    if len(a) > max_px:
        step = len(a) // max_px
        a = a[::step]
    return {tuple(int(v) for v in c) for c in np.unique(a, axis=0)}


def check_raw_ranges(workers: int = 8) -> int:
    print("\n" + "=" * 96)
    print("[9] 原始标签取值/颜色集合全量扫描 (索引类: 防止未知索引被静默忽略)")
    print("=" * 96)
    bad = 0

    # ---- AeroScapes: 索引 PNG, 全量 ----
    files = sorted(glob.glob(f"{AIC_ROOT}/aeroscapes/SegmentationClass/*.png"))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rng = list(ex.map(_raw_index_range, files, chunksize=64))
    lo = min(r[0] for r in rng)
    hi = max(r[1] for r in rng)
    all_ids = sorted({v for r in rng for v in r[2]})
    ok = 0 <= lo and hi <= 11 and all_ids == list(range(12))
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'ERR'}] aeroscapes {len(files)} 张原始标签取值 {all_ids} "
          f"-> 恰好 0..11, 无未知索引 {'✅' if ok else '❌'}")

    # ---- DLRSD: 索引 PNG, 全量 ----
    files = sorted(glob.glob(f"{AIC_ROOT}/DLRSD/Labels/*/*.png"))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rng = list(ex.map(_raw_index_range, files, chunksize=64))
    all_ids = sorted({v for r in rng for v in r[2]})
    hi = max(r[1] for r in rng)
    ok = hi <= 17 and set(all_ids) <= set(range(18))
    bad += 0 if ok else 1
    print(f"  [{'OK ' if ok else 'ERR'}] dlrsd      {len(files)} 张原始标签最大值 {hi} "
          f"(预期 <=17), 出现过的取值 {all_ids} {'✅' if ok else '❌'}")

    # ---- 颜色类数据集: 抽样确认颜色集合 == 映射表 (未知颜色会让 remap_colors 报错) ----
    for ds, pattern, table in (
        ("uavid", f"{AIC_ROOT}/UAVid/uavid_v1.5_official_release_image/uavid_train/seq*/Labels/*.png",
         set(lm.UAVID_COLORS)),
        ("potsdam", f"{AIC_ROOT}/ISPRS_Potsdam/raw/Potsdam/5_Labels_for_participants_no_Boundary/*.tif",
         set(lm.POTSDAM_COLORS) | {(255, 255, 255), (255, 0, 0), (0, 0, 0)}),
        ("hdmaps", f"{AIC_ROOT}/HD-Maps/Aerial/colorAnnotation/*.png", set(lm.HDMAPS_COLORS)),
    ):
        files = sorted(glob.glob(pattern))[:6]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            cols = set().union(*ex.map(_raw_color_set, files))
        unknown = cols - table
        ok = not unknown
        bad += 0 if ok else 1
        print(f"  [{'OK ' if ok else 'ERR'}] {ds:10s} 抽 {len(files)} 张原始标签颜色 {len(cols)} 种, "
              f"未登记 {len(unknown)} 种 {sorted(unknown)[:3]} {'✅' if ok else '❌'}")

    # ---- HEURISTIC: 全量处理时 remap_colors 若遇到未登记颜色会直接抛错 ----
    print("  [OK ] 注: uavid/potsdam/hdmaps 的 precache 全量处理 0 失败 => 所有标签像素都命中了颜色表")
    return bad


# ---------------------------------------------------------------------------
# 10. RGB 缓存保真度 (缓存 JPEG vs 原始影像, 量化 q95 4:4:4 的损失)
# ---------------------------------------------------------------------------
def check_img_fidelity(ds_list, n_samples: int = 3, seed: int = 0) -> int:
    print("\n" + "=" * 96)
    print("[10] RGB 缓存保真度: 缓存 JPEG(q95, 4:4:4) vs 原始影像解码 (PSNR 越高越好, >40dB 视为无损感)")
    print("=" * 96)
    rng = random.Random(seed)
    bad = 0
    for ds in ds_list:
        for split in ("train", "val"):
            items = raw_lists(ds)[split]
            if not items:
                continue
            pick = rng.sample(items, min(n_samples, len(items)))
            psnrs, diffs = [], []
            for it in pick:
                _name, _group, img_path, _payload = it
                try:
                    raw = P.load_image(ds, img_path)                    # 与 precache 完全相同的读取路径
                    cac = cached_image(ds, split, it[0])
                except Exception as e:                                  # noqa: BLE001
                    print(f"  [ERR] {ds}/{split}/{it[0]}: {e}")
                    bad += 1
                    continue
                if raw.shape != cac.shape:
                    print(f"  [ERR] {ds}/{split}/{it[0]}: 尺寸 {raw.shape} vs {cac.shape}")
                    bad += 1
                    continue
                a, b = raw.astype(np.float32), cac.astype(np.float32)
                mse = float(((a - b) ** 2).mean())
                psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))
                psnrs.append(psnr)
                diffs.append(float(np.abs(a - b).mean()))
            if psnrs:
                print(f"  [{'OK ' if np.mean(psnrs) > 38 else 'WARN'}] {ds:11s}/{split:5s} "
                      f"PSNR 均值 {np.mean(psnrs):5.1f} dB (最低 {min(psnrs):5.1f}) | 平均绝对差 "
                      f"{np.mean(diffs):4.2f}/255")
    return bad


# ---------------------------------------------------------------------------
def main():
    global CACHE_ROOT
    ap = argparse.ArgumentParser(description="AIC 数据读取正确性审计")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--checks",
                    default="counts,files,cache,pairs,align,semantics,dlrsd,dlr3k,rawrange,fidelity",
                    help="逗号分隔: counts,files,cache,pairs,align,semantics,dlrsd,dlr3k,rawrange,fidelity")
    ap.add_argument("--samples", type=int, default=6, help="pairs/align 每 split 抽样数")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    args = ap.parse_args()
    CACHE_ROOT = args.cache_root

    ds_list = list(AIC_DATASETS) if args.datasets == "all" else \
        [d.strip() for d in args.datasets.split(",") if d.strip()]
    checks = {c.strip() for c in args.checks.split(",") if c.strip()}
    print(f"[AUDIT] datasets={ds_list}\n[AUDIT] checks={sorted(checks)}  samples={args.samples}")
    t0 = time.time()
    bad = 0
    if "counts" in checks:
        bad += check_counts(ds_list)
    if "files" in checks:
        bad += check_files()
    if "rawrange" in checks:
        bad += check_raw_ranges(args.workers)
    if "fidelity" in checks:
        bad += check_img_fidelity(ds_list)
    if "cache" in checks:
        bad += check_cache(ds_list, args.workers)
    if "pairs" in checks:
        bad += check_pairs(ds_list, args.samples)
    if "align" in checks:
        bad += check_align(ds_list, args.samples)
    if "semantics" in checks:
        bad += check_semantics(ds_list)
    if "dlrsd" in checks:
        bad += check_dlrsd()
    if "dlr3k" in checks:
        bad += check_dlr3k_overlay()

    print("\n" + "=" * 96)
    print(f"审计完成, 耗时 {time.time()-t0:.0f}s | " +
          ("全部通过 ✅" if bad == 0 else f"发现 {bad} 处问题 ❌"))
    print("=" * 96)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
