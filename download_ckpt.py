# -*- coding: utf-8 -*-
"""
下载 StreamVGGT 预训练权重 (HuggingFace: lch01/StreamVGGT)

用法:
    python download_ckpt.py                 # -> ../StreamVGGT/ckpt/checkpoints.pth
    python download_ckpt.py --out-dir ckpt  # 自定义目录
    HF_ENDPOINT=https://hf-mirror.com python download_ckpt.py   # 国内镜像

权重: checkpoints.pth (5.03 GB) — StreamVGGT 主干 (aggregator + camera/depth/point/track head)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ID = "lch01/StreamVGGT"
FILES = ["checkpoints.pth", "config.json"]
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "StreamVGGT" / "ckpt"


def main():
    ap = argparse.ArgumentParser(description="下载 StreamVGGT 预训练权重")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--repo-id", default=REPO_ID)
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERR] 缺少 huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        return 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_hf_tmp"
    for fn in FILES:
        dst = out / fn
        if dst.exists() and dst.stat().st_size > 1_000_000:
            print(f"[SKIP] {dst} 已存在 ({dst.stat().st_size/1e9:.2f} GB)")
            continue
        t0 = time.time()
        print(f"[GET ] {args.repo_id}/{fn} ...", flush=True)
        p = hf_hub_download(repo_id=args.repo_id, filename=fn, local_dir=str(tmp))
        shutil.move(p, str(dst))
        print(f"[OK  ] {dst} ({dst.stat().st_size/1e9:.3f} GB, {time.time()-t0:.0f}s)")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[DONE] 权重目录: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
