#!/usr/bin/env python3
"""
快速扫描 v3 — 用标准引擎短训 + 对抗筛选。

v3 修复：
- 使用实例化 ModelConfig，避免全局状态污染
- 保留最优公式时使用新的 composite 标准
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from model_core.config import ModelConfig

SWEEPS = [
    {"train_steps": 500, "batch_size": 4096, "learning_rate": 1e-3, "desc": "baseline"},
    {"train_steps": 500, "batch_size": 8192, "learning_rate": 1e-3, "desc": "large batch"},
    {"train_steps": 800, "batch_size": 4096, "learning_rate": 1e-3, "desc": "more steps"},
]


def screen_formula(formula, loader, train_feat, train_target, val_feat, val_target):
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
        desc = f"{args.inst_id} {params['desc']} (bs={params['batch_size']} steps={params['train_steps']})"
        print(f"\n[{desc}]")

        # 创建隔离的配置实例
        config = ModelConfig(
            train_steps=params["train_steps"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            multi_dim_fitness=False,
            adversarial_trials=0,
        )

        from model_core.engine import AlphaEngine
        engine = AlphaEngine(
            config=config,
            inst_id=args.inst_id,
            bar="1H",
            candle_limit=2000,
            use_lord_regularization=False,
        )
        # 复用已加载的数据
        engine.loader = loader
        engine.train_feat = train_feat.to(config.device)
        engine.train_target = train_target.to(config.device)
        engine.val_feat = val_feat.to(config.device)
        engine.val_target = val_target.to(config.device)
        engine.train()

        best_formula = engine.best_formula
        train_ic = engine.best_train_ic

        if best_formula is None:
            print(f"  => No formula found")
            continue

        adv_train, adv_val, adv_ret = screen_formula(
            best_formula, loader, train_feat, train_target, val_feat, val_target
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

    # 汇总
    results.sort(key=lambda x: x["val_adv"] if x["val_adv"] is not None else -999, reverse=True)
    print(f"\n{'='*70}")
    print(f"  SWEEP RESULTS: {args.inst_id}")
    print(f"{'='*70}")
    print(f"  {'Desc':<20} {'IC':>8} {'AdvTr':>8} {'AdvVal':>8} {'Ret':>8} {'Uniq':>5}")
    print(f"  {'-'*55}")
    for r in results:
        marker = "[KEEP]" if r["val_adv"] > 0 else "[DROP]"
        print(f"  {r['desc']:<20} {r['train_ic']:>8.4f} {r['train_adv']:>8.2f} {r['val_adv']:>8.2f} {r['val_ret']:>7.2%} {r['n_unique']:>5}  {marker}")

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
        print(f"\n  No formula passed filter.")
