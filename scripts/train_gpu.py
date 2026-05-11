#!/usr/bin/env python3
"""
GPU 一键训练脚本。

环境变量（.env 或命令行 export）：
    D_MODEL=128 N_LAYER=4 BATCH_SIZE=65536 TRAIN_STEPS=2000

用法:
    # 基础训练
    python scripts/train_gpu.py

    # 大模型 + oracle 预训练
    D_MODEL=256 N_LAYER=4 BATCH_SIZE=131072 TRAIN_STEPS=5000 \
        python scripts/train_gpu.py --pretrain-oracle --inst-id ETH-USDT

    # 仅 oracle benchmark（不训练）
    python scripts/train_gpu.py --oracle-only --max-len 8
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine


def main():
    parser = argparse.ArgumentParser(description="GPU Training Script")
    parser.add_argument("--inst-id", default="BTC-USDT")
    parser.add_argument("--bar", default="1H")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--pretrain-oracle", action="store_true")
    parser.add_argument("--oracle-max-len", type=int, default=6)
    parser.add_argument("--oracle-topk", type=int, default=500)
    parser.add_argument("--oracle-only", action="store_true", help="Run oracle benchmark then exit")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    # 打印 GPU 信息
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        logger.info(f"[GPU] {gpu.name} {gpu.total_memory / 1024**3:.1f}GB")
        logger.info(f"[GPU] CUDA {torch.version.cuda} | PyTorch {torch.__version__}")
    else:
        logger.warning("[GPU] CUDA not available, training on CPU")

    config = ModelConfig(
        use_amp=not args.no_amp,
        compile_model=not args.no_compile,
    )
    logger.info(f"[Config] d_model={config.d_model} n_layer={config.n_layer} n_head={config.n_head}")
    logger.info(f"[Config] batch={config.batch_size} accum={config.grad_accum_steps} effective={config.batch_size * config.grad_accum_steps}")
    logger.info(f"[Config] amp={config.use_amp} compile={config.compile_model}")

    if args.oracle_only:
        from scripts.benchmark_oracle import main as oracle_main
        import sys as _sys
        _sys.argv = [
            "benchmark_oracle.py",
            args.inst_id,
            "--bar", args.bar,
            "--limit", str(args.limit),
            "--max-len", str(args.oracle_max_len),
            "--topk", str(args.oracle_topk),
        ]
        return oracle_main()

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
    logger.info(f"[Done] Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"[Done] Best formula saved to {config.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
