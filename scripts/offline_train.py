#!/usr/bin/env python3
"""
离线训练脚本 —— 适用于无外网环境的 GPU 服务器。

不访问 OKX API，只读取本地 CSV + meta.json。

用法:
    # 1. 先在 macOS 上拉数据并打包
    #    python fetch_cache.py
    #    tar czf bundle_in.tar.gz data_cache/
    #    scp bundle_in.tar.gz gpu_server:/path/to/AlphaGPT/

    # 2. 在 GPU 服务器上解压并训练
    #    tar xzf bundle_in.tar.gz
    #    python scripts/offline_train.py --bundle-in data_cache/

    # 3. 训练完成后打包结果传回 macOS
    #    python scripts/offline_train.py --bundle-in data_cache/ --bundle-out results.tar.gz
    #    scp results.tar.gz macos:/path/to/AlphaGPT/output/
"""

import argparse
import os
import sys
import tarfile
import time
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine


def bundle_output(save_dir: str, inst_id: str, bar: str, output_tar: str):
    """将训练结果打包为 tar.gz。"""
    prefix = inst_id.replace("-", "")
    files_to_pack = []
    save_path = Path(save_dir)

    for pattern in [f"{prefix}_{bar}_formula.json", f"{prefix}_{bar}_history.json"]:
        fp = save_path / pattern
        if fp.exists():
            files_to_pack.append(fp)

    model_ckpt = save_path / f"{prefix}_{bar}_model.pt"
    if model_ckpt.exists():
        files_to_pack.append(model_ckpt)

    if not files_to_pack:
        logger.warning(f"No output files found in {save_dir}")
        return

    with tarfile.open(output_tar, "w:gz") as tar:
        for fp in files_to_pack:
            tar.add(fp, arcname=fp.name)

    logger.info(f"[Bundle] Packed {len(files_to_pack)} files into {output_tar}")
    for fp in files_to_pack:
        logger.info(f"  - {fp.name}")


def main():
    parser = argparse.ArgumentParser(description="Offline Training (no network)")
    parser.add_argument("--inst-id", default="BTC-USDT")
    parser.add_argument("--bar", default="1H")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--cache-dir", default="data_cache", help="Local CSV cache directory")
    parser.add_argument("--pretrain-oracle", action="store_true")
    parser.add_argument("--oracle-max-len", type=int, default=6)
    parser.add_argument("--oracle-topk", type=int, default=500)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--bundle-out", default="", help="Output tar.gz path after training")
    args = parser.parse_args()

    cache_path = Path(args.cache_dir)
    csv_file = cache_path / f"{args.inst_id.replace('-', '')}_{args.bar}.csv"

    if not csv_file.exists():
        logger.error(f"CSV not found: {csv_file}")
        logger.error("Please fetch data on a machine with network access and copy to this directory.")
        sys.exit(1)

    # 检查 meta.json
    meta_file = cache_path / f"{args.inst_id.replace('-', '')}_{args.bar}.meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        logger.info(f"[Data] {meta['inst_id']} {meta['bar']} {meta['rows']} rows")
        logger.info(f"[Data] Range: {meta['ts_min']} → {meta['ts_max']}")
    else:
        logger.warning(f"[Data] No meta.json found for {args.inst_id}")

    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        logger.info(f"[GPU] {gpu.name} {gpu.total_memory / 1024**3:.1f}GB")
    else:
        logger.warning("[GPU] CUDA not available, training on CPU")

    config = ModelConfig(
        use_amp=not args.no_amp,
        compile_model=not args.no_compile,
    )
    logger.info(f"[Config] d_model={config.d_model} n_layer={config.n_layer}")
    logger.info(f"[Config] batch={config.batch_size} amp={config.use_amp} compile={config.compile_model}")

    t0 = time.time()
    engine = AlphaEngine(
        config=config,
        inst_id=args.inst_id,
        bar=args.bar,
        candle_limit=args.limit,
        use_lord_regularization=True,
        pretrain_oracle=args.pretrain_oracle,
        oracle_max_len=args.oracle_max_len,
        oracle_topk=args.oracle_topk,
    )
    engine.train()

    elapsed = time.time() - t0
    logger.info(f"[Done] Training time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # 打包输出
    if args.bundle_out:
        bundle_output(config.save_dir, args.inst_id, args.bar, args.bundle_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
