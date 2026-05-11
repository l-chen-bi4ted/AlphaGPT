#!/usr/bin/env python3
"""
Oracle Benchmark — 穷举搜索建立 RL 性能上界。

用法:
    python scripts/benchmark_oracle.py BTC-USDT --max-len 6 --ops basic --topk 50
    python scripts/benchmark_oracle.py BTC-USDT --max-len 8 --ops extended --topk 100 --device cuda

输出:
    output/<inst_id>_oracle_L<max_len>.json   # top-k 公式
    output/<inst_id>_oracle_landscape.png      # IC 景观图（可选 matplotlib）
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from model_core.config import ModelConfig
from model_core.oracle import ExhaustiveOracle
from okx_data import OKXDataLoader


def main():
    parser = argparse.ArgumentParser(description="Exhaustive Oracle Benchmark")
    parser.add_argument("inst_id", nargs="?", default="BTC-USDT", help="OKX trading pair")
    parser.add_argument("--bar", default="1H", help="K-line interval")
    parser.add_argument("--limit", type=int, default=2000, help="Candle count")
    parser.add_argument("--max-len", type=int, default=6, help="Max formula length (recommend ≤8)")
    parser.add_argument("--min-len", type=int, default=None, help="Min formula length (default=max_len-2)")
    parser.add_argument("--ops", default="basic", choices=["basic", "extended", "all"], help="Operator subset")
    parser.add_argument("--k-feats", type=int, default=6, help="Number of features to use")
    parser.add_argument("--topk", type=int, default=50, help="Top-k formulas to save")
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--val-split", type=float, default=0.2, help="OOS validation split")
    parser.add_argument("--no-val", action="store_true", help="Skip validation, train-only")
    parser.add_argument("--compare-rl", action="store_true", help="Compare with a quick RL run")
    parser.add_argument("--plot", action="store_true", help="Generate IC landscape plot (needs matplotlib)")
    args = parser.parse_args()

    config = ModelConfig(device=args.device)
    min_len = args.min_len or max(3, args.max_len - 2)

    logger.info(f"[Oracle Benchmark] {args.inst_id} {args.bar}")
    logger.info(f"  max_len={args.max_len} min_len={min_len} ops={args.ops} topk={args.topk}")

    # 加载数据
    loader = OKXDataLoader(args.inst_id, args.bar, args.limit, config=config)
    loader.load_data()
    logger.info(f"  Data loaded: {loader.feat_tensor.shape}")

    # 创建 Oracle
    oracle = ExhaustiveOracle(
        config=config,
        max_len=args.max_len,
        min_len=min_len,
        ops_subset=args.ops,
        k_feats=args.k_feats,
    )

    # 运行搜索
    t0 = time.time()
    results = oracle.search(
        loader,
        topk=args.topk,
        val_split=0.0 if args.no_val else args.val_split,
        verbose=True,
    )
    elapsed = time.time() - t0

    if not results:
        logger.error("Oracle found no valid formulas!")
        return 1

    # 保存结果
    os.makedirs(config.save_dir, exist_ok=True)
    prefix = f"{args.inst_id.replace('-', '')}_{args.bar}"
    out_path = os.path.join(config.save_dir, f"{prefix}_oracle_L{args.max_len}.json")

    save_data = {
        "inst_id": args.inst_id,
        "bar": args.bar,
        "max_len": args.max_len,
        "min_len": min_len,
        "ops_subset": args.ops,
        "elapsed_sec": elapsed,
        "total_evaluated": len(results),
        "topk": [
            {
                "rank": i + 1,
                "formula": r.formula,
                "formula_str": r.formula_str,
                "train_ic": round(r.train_ic, 6),
                "val_ic": round(r.val_ic, 6) if r.val_ic is not None else None,
                "composite": round(r.composite, 6),
                "n_ops": r.n_ops,
            }
            for i, r in enumerate(results)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"[Oracle] Results saved to {out_path}")

    # 打印摘要
    best = results[0]
    logger.info(f"[Oracle] Best formula: {best.formula_str}")
    logger.info(f"[Oracle]   train_ic={best.train_ic:.4f} val_ic={best.val_ic:.4f if best.val_ic else 'N/A'} composite={best.composite:.4f}")
    logger.info(f"[Oracle]   ops={best.n_ops} len={len(best.formula)}")

    # 与 RL 对比
    if args.compare_rl:
        logger.info("[Oracle] Running quick RL comparison...")
        from model_core.engine import AlphaEngine
        rl_config = ModelConfig(
            train_steps=200,
            batch_size=2048,
            device=args.device,
        )
        engine = AlphaEngine(
            config=rl_config,
            inst_id=args.inst_id,
            bar=args.bar,
            candle_limit=args.limit,
            use_lord_regularization=False,
        )
        # 复用 loader
        engine.loader = loader
        engine.train_feat = oracle.train_feat
        engine.train_target = oracle.train_target
        engine.val_feat = oracle.val_feat
        engine.val_target = oracle.val_target
        engine.train()

        logger.info(f"[RL] Best train_ic={engine.best_train_ic:.4f} val_ic={engine.best_val_ic:.4f}")
        logger.info(f"[RL] vs Oracle train_ic={best.train_ic:.4f} val_ic={best.val_ic:.4f if best.val_ic else 'N/A'}")

        gap = (best.train_ic - engine.best_train_ic) / abs(best.train_ic + 1e-6) * 100
        logger.info(f"[Oracle-RL Gap] RL achieves {100 - gap:.1f}% of oracle performance")

    # 绘图
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # 左图: formula length vs IC
            lengths = [len(r.formula) for r in results]
            ics = [r.train_ic for r in results]
            axes[0].scatter(lengths, ics, alpha=0.6, s=20)
            axes[0].set_xlabel("Formula Length")
            axes[0].set_ylabel("Train IC")
            axes[0].set_title(f"IC vs Formula Length (L≤{args.max_len})")
            axes[0].grid(True, alpha=0.3)

            # 右图: ops count vs IC
            ops_counts = [r.n_ops for r in results]
            axes[1].scatter(ops_counts, ics, alpha=0.6, s=20, c="orange")
            axes[1].set_xlabel("Operator Count")
            axes[1].set_ylabel("Train IC")
            axes[1].set_title("IC vs Operator Count")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(config.save_dir, f"{prefix}_oracle_landscape.png")
            plt.savefig(plot_path, dpi=150)
            logger.info(f"[Oracle] Plot saved to {plot_path}")
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot")

    return 0


if __name__ == "__main__":
    sys.exit(main())
