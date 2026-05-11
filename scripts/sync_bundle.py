#!/usr/bin/env python3
"""
跨机器工作流同步工具。

场景：macOS 拉数据 → GPU 服务器训练 → macOS 跑模拟盘

打包（macOS 上执行，准备传给 GPU 服务器）:
    python scripts/sync_bundle.py pack-in data_cache/ bundle_in.tar.gz
    # 输出: bundle_in.tar.gz (含 CSV + meta.json)

解包（GPU 服务器上执行，接收数据）:
    python scripts/sync_bundle.py unpack-in bundle_in.tar.gz
    # 解压到 data_cache/

打包（GPU 服务器训练后，传回 macOS）:
    python scripts/sync_bundle.py pack-out output/ bundle_out.tar.gz --inst-id BTC-USDT --bar 1H
    # 输出: bundle_out.tar.gz (含 formula.json + history.json + model.pt)

解包（macOS 上执行，接收训练结果）:
    python scripts/sync_bundle.py unpack-out bundle_out.tar.gz output/
    # 解压到 output/
"""

import argparse
import os
import sys
import tarfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger


def pack_data(cache_dir: str, output_tar: str):
    """打包数据缓存（CSV + meta）。"""
    cache = Path(cache_dir)
    if not cache.exists():
        logger.error(f"Cache dir not found: {cache}")
        sys.exit(1)

    files = list(cache.glob("*.csv")) + list(cache.glob("*.meta.json"))
    if not files:
        logger.error(f"No CSV or meta files found in {cache}")
        sys.exit(1)

    with tarfile.open(output_tar, "w:gz") as tar:
        for fp in files:
            tar.add(fp, arcname=f"data_cache/{fp.name}")

    logger.info(f"[Pack] {len(files)} files → {output_tar}")
    for fp in files:
        logger.info(f"  - {fp.name}")


def unpack_data(input_tar: str, target_dir: str = "data_cache"):
    """解压数据缓存。"""
    if not os.path.exists(input_tar):
        logger.error(f"Tar file not found: {input_tar}")
        sys.exit(1)

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    with tarfile.open(input_tar, "r:gz") as tar:
        tar.extractall(path=target_dir if target_dir != "." else "")

    logger.info(f"[Unpack] {input_tar} → {target_dir}")


def pack_results(save_dir: str, output_tar: str, inst_id: str, bar: str):
    """打包训练结果。"""
    prefix = inst_id.replace("-", "")
    save = Path(save_dir)

    files = []
    for pattern in [
        f"{prefix}_{bar}_formula.json",
        f"{prefix}_{bar}_history.json",
        f"{prefix}_{bar}_model.pt",
        f"{prefix}_{bar}_sweep_best.json",
    ]:
        fp = save / pattern
        if fp.exists():
            files.append(fp)

    if not files:
        logger.error(f"No result files found in {save_dir} for {inst_id} {bar}")
        sys.exit(1)

    with tarfile.open(output_tar, "w:gz") as tar:
        for fp in files:
            tar.add(fp, arcname=f"output/{fp.name}")

    logger.info(f"[Pack] {len(files)} files → {output_tar}")
    for fp in files:
        logger.info(f"  - {fp.name}")


def unpack_results(input_tar: str, target_dir: str = "output"):
    """解压训练结果。"""
    if not os.path.exists(input_tar):
        logger.error(f"Tar file not found: {input_tar}")
        sys.exit(1)

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    with tarfile.open(input_tar, "r:gz") as tar:
        tar.extractall(path=target_dir if target_dir != "." else "")

    logger.info(f"[Unpack] {input_tar} → {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="Cross-machine sync bundler")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_in = sub.add_parser("pack-in", help="Pack data_cache for GPU server")
    p_in.add_argument("cache_dir", default="data_cache")
    p_in.add_argument("output_tar", default="bundle_in.tar.gz")

    p_uin = sub.add_parser("unpack-in", help="Unpack data on GPU server")
    p_uin.add_argument("input_tar", default="bundle_in.tar.gz")
    p_uin.add_argument("--target-dir", default="data_cache")

    p_out = sub.add_parser("pack-out", help="Pack training results for macOS")
    p_out.add_argument("save_dir", default="output")
    p_out.add_argument("output_tar", default="bundle_out.tar.gz")
    p_out.add_argument("--inst-id", default="BTC-USDT")
    p_out.add_argument("--bar", default="1H")

    p_uout = sub.add_parser("unpack-out", help="Unpack results on macOS")
    p_uout.add_argument("input_tar", default="bundle_out.tar.gz")
    p_uout.add_argument("--target-dir", default="output")

    args = parser.parse_args()

    if args.cmd == "pack-in":
        pack_data(args.cache_dir, args.output_tar)
    elif args.cmd == "unpack-in":
        unpack_data(args.input_tar, args.target_dir)
    elif args.cmd == "pack-out":
        pack_results(args.save_dir, args.output_tar, args.inst_id, args.bar)
    elif args.cmd == "unpack-out":
        unpack_results(args.input_tar, args.target_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
