#!/usr/bin/env python3
"""
快速扫描 v3 — 用标准引擎短训 + 对抗筛选。

思路：训练用简单 reward（快速收敛），训练后用对抗 eval 筛。
      之前的问题是训练本身引入对抗噪声，302 步根本不够收敛。

用法:
    python sweep.py
    python sweep.py ETH-USDT
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from model_core.config import ModelConfig

SWEEPS = [
    {"TRAIN_STEPS": 500, "BATCH_SIZE": 4096, "LEARNING_RATE": 1e-3, "desc": "baseline"},
    {"TRAIN_STEPS": 500, "BATCH_SIZE": 8192, "LEARNING_RATE": 1e-3, "desc": "large batch"},
    {"TRAIN_STEPS": 800, "BATCH_SIZE": 4096, "LEARNING_RATE": 1e-3, "desc": "more steps"},
]


def screen_formula(formula, inst_id, bar, loader, train_feat, train_target, val_feat, val_target):
    """对抗 + 多维评估，返回 train_score, val_score, val_ret。"""
    from model_core.vm import StackVM
    from model_core.backtest import CEXBacktest

    vm = StackVM()
    res_train = vm.execute(formula, train_feat)
    res_val = vm.execute(formula, val_feat)
    if res_train is None or res_val is None:
        return None, None, None

    bt = CEXBacktest(adversarial_trials=3, noise_std=0.02, multi_dim=True)
    train_score, _ = bt.evaluate(res_train, loader.raw_data_cache, train_target)
    val_score, val_ret = bt.evaluate(res_val, loader.raw_data_cache, val_target)
    return train_score.item(), val_score.item(), val_ret


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("inst_id", nargs="?", default="BTC-USDT")
    args = parser.parse_args()

    # ── 加载数据 ──
    from okx_data import OKXDataLoader
    loader = OKXDataLoader(args.inst_id, "1H", 2000)
    loader.load_data()
    T = loader.feat_tensor.shape[-1]
    split = int(T * 0.8)
    train_feat = loader.feat_tensor[..., :split]
    train_target = loader.target_ret[..., :split]
    val_feat = loader.feat_tensor[..., split:]
    val_target = loader.target_ret[..., split:]
    print(f"Data: {T} candles, train={split} val={T-split}")

    results = []

    for params in SWEEPS:
        desc = f"{args.inst_id} {params['desc']} (bs={params['BATCH_SIZE']} steps={params['TRAIN_STEPS']})"
        print(f"\n[{desc}]")

        # ── 用 AlphaEngine 标准训练（简单 reward，快速收敛）──
        ModelConfig.BATCH_SIZE = params["BATCH_SIZE"]
        ModelConfig.TRAIN_STEPS = params["TRAIN_STEPS"]
        ModelConfig.LEARNING_RATE = params["LEARNING_RATE"]
        ModelConfig.MULTI_DIM_FITNESS = False  # ← 训练不用多维
        ModelConfig.ADVERSARIAL_TRIALS = 0     # ← 训练不加对抗

        from model_core.engine import AlphaEngine
        engine = AlphaEngine(
            inst_id=args.inst_id, bar="1H",
            candle_limit=2000,
            use_lord_regularization=False,
        )
        # Override the loader's data with our split
        engine.loader = loader
        engine.train_feat = train_feat
        engine.train_target = train_target
        engine.val_feat = val_feat
        engine.val_target = val_target
        engine.train()

        best_formula = engine.best_formula
        train_ic = engine.best_score  # Now IC, not backtest score

        # ── 对抗筛选 ──
        if best_formula is None:
            print(f"  => No formula found")
            continue

        adv_train, adv_val, adv_ret = screen_formula(
            best_formula, args.inst_id, "1H",
            loader, train_feat, train_target, val_feat, val_target
        )

        if adv_val is not None:
            n_unique = len(set(best_formula))
            results.append({
                "desc": params["desc"],
                "formula": best_formula,
                "train_ic": train_ic,
                "train_adv": adv_train,
                "val_adv": adv_val,
                "val_ret": adv_ret,
                "n_unique": n_unique,
            })
            marker = "[KEEP]" if adv_val > 0 else "[DROP]"
            print(f"  => IC={train_ic:.4f}  adv_train={adv_train:.2f}  adv_val={adv_val:.2f}  ret={adv_ret:.2%}  {marker}")

    # ── 汇总 ──
    results.sort(key=lambda x: x["val_adv"], reverse=True)
    print(f"\n{'='*70}")
    print(f"  SWEEP RESULTS: {args.inst_id}")
    print(f"{'='*70}")
    print(f"  {'Desc':<20} {'IC':>8} {'AdvTr':>8} {'AdvVal':>8} {'Ret':>8} {'Uniq':>5}")
    print(f"  {'-'*55}")
    for r in results:
        marker = "[KEEP]" if r["val_adv"] > 0 else "[DROP]"
        print(f"  {r['desc']:<20} {r['train_ic']:>8.4f} {r['train_adv']:>8.2f} {r['val_adv']:>8.2f} {r['val_ret']:>7.2%} {r['n_unique']:>5}  {marker}")

    # ── 保存最优 ──
    best = results[0] if results else None
    if best and best["val_adv"] > 0:
        os.makedirs("output", exist_ok=True)
        name = args.inst_id.replace("-", "")
        fp = f"output/{name}_1H_sweep_best.json"
        with open(fp, "w") as f:
            json.dump({
                "inst_id": args.inst_id, "bar": "1H",
                "train_score": best["train_adv"], "val_score": best["val_adv"],
                "formula": best["formula"],
            }, f, indent=2)
        print(f"\n  Best saved: {fp}")
    else:
        print(f"\n  No formula passed filter. Consider different data or more steps.")
