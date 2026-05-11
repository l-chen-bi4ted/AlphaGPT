#!/usr/bin/env python3
"""
公式审计工具 —— 对已训练的公式做样本外验证。

用法:
    python eval_formula.py                                    # 默认评估 output/BTCUSDT_1H_formula.json
    python eval_formula.py ETH-USDT 1H                        # 指定品种
    python eval_formula.py BTC-USDT 1H --adversarial 10       # 对抗模式压测
    python eval_formula.py --all                              # 评估 output/ 下所有公式

输出:
    train_score / val_score / overfit 警告 / 复杂度报告
"""

import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from okx_data import OKXDataLoader
from model_core.vm import StackVM
from model_core.backtest import CEXBacktest
from model_core.config import ModelConfig, default_config
from model_core.ops import OPS_CONFIG


# ─── 算子名称映射 ──────────────────────────────────
OP_NAMES = {
    0: "RET", 1: "LIQ", 2: "PRESSURE", 3: "FOMO", 4: "DEV", 5: "LOG_VOL",
}
for i, op in enumerate(OPS_CONFIG):
    OP_NAMES[6 + i] = op[0]


def decode_formula(formula: list) -> str:
    """将公式 token 序列翻译为可读字符串。"""
    return " → ".join(OP_NAMES.get(t, f"?{t}") for t in formula)


def formula_complexity(formula: list) -> dict:
    """计算公式复杂度指标。"""
    n_unique = len(set(formula))
    n_ops = sum(1 for t in formula if t >= 6)
    n_factors = len(formula) - n_ops

    # 统计每个算子出现次数
    op_counts = {}
    for t in formula:
        name = OP_NAMES.get(t, f"?{t}")
        op_counts[name] = op_counts.get(name, 0) + 1

    # 复杂度分数（0~1，越低越好）
    score = min((n_unique - 3) / 10, 1.0)  # 3种以内不扣分

    return {
        "n_unique": n_unique,
        "n_ops": n_ops,
        "n_factors": n_factors,
        "composition": op_counts,
        "complexity_score": round(1 - score, 2),
    }


def evaluate_formula(
    formula: list,
    inst_id: str = "BTC-USDT",
    bar: str = "1H",
    split_ratio: float = 0.8,
    adversarial_trials: int = 0,
    noise_std: float = 0.02,
) -> dict:
    """
    加载数据 → 执行公式 → 样本外评估 → 返回完整报告。
    """
    print(f"\n{'='*60}")
    print(f"  Formula Audit: {inst_id} {bar}")
    print(f"{'='*60}")

    # 1. 加载数据
    loader = OKXDataLoader(inst_id, bar, limit=2000)
    loader.load_data()
    feat = loader.feat_tensor  # [1, 6, T]
    target = loader.target_ret  # [1, T]
    T = feat.shape[-1]
    print(f"  数据: {T} 根 K 线")

    # 2. 时间序列 split
    split_idx = int(T * split_ratio)
    train_feat = feat[..., :split_idx]
    train_target = target[..., :split_idx]
    val_feat = feat[..., split_idx:]
    val_target = target[..., split_idx:]
    print(f"  Split: train={split_idx} val={T - split_idx}  (ratio={split_ratio})")

    # 3. 执行公式
    vm = StackVM()
    res_train = vm.execute(formula, train_feat)
    res_val = vm.execute(formula, val_feat)

    if res_train is None:
        return {"error": "公式在训练集执行失败"}
    if res_val is None:
        return {"error": "公式在验证集执行失败"}

    # 4. 评估
    bt = CEXBacktest(
        adversarial_trials=adversarial_trials,
        noise_std=noise_std,
        multi_dim=True,
    )
    raw = loader.raw_data_cache

    train_result, train_ret = bt.evaluate(res_train, raw, train_target)
    val_result, val_ret = bt.evaluate(res_val, raw, val_target)

    train_score = train_result.item()
    val_score = val_result.item()

    # 5. 复杂度
    cx = formula_complexity(formula)

    # 6. 过拟合诊断
    overfit = False
    overfit_pct = 0
    if val_score > -900 and train_score > 0.01:
        overfit_pct = val_score / train_score * 100
        if val_score < train_score * 0.3:
            overfit = True

    return {
        "inst_id": inst_id,
        "bar": bar,
        "formula": formula,
        "decoded": decode_formula(formula),
        "train_score": train_score,
        "val_score": val_score,
        "train_return": train_ret,
        "val_return": val_ret,
        "overfit": overfit,
        "overfit_pct": overfit_pct,
        "complexity": cx,
        "adversarial_trials": adversarial_trials,
    }


def print_report(r: dict):
    """格式化打印审计报告。"""
    if "error" in r:
        print(f"\n  [ERROR] {r['error']}")
        return

    print(f"\n  [Formula] {r['decoded']}")
    print(f"  [Complexity] {r['complexity']['complexity_score']:.2f}  "
          f"(unique_ops={r['complexity']['n_unique']}, "
          f"factors={r['complexity']['n_factors']}, "
          f"ops={r['complexity']['n_ops']})")

    print(f"\n  {'-'*40}")
    print(f"  {'Metric':<15} {'Train':>12} {'Val (OOS)':>12}")
    print(f"  {'-'*40}")
    print(f"  {'Score':<15} {r['train_score']:>12.4f} {r['val_score']:>12.4f}")
    print(f"  {'Return':<15} {r['train_return']:>11.2%} {r['val_return']:>11.2%}")

    if r["adversarial_trials"] > 0:
        print(f"\n  [!] Adversarial mode: {r['adversarial_trials']} trials x noise 0.02")

    if r["overfit"]:
        print(f"\n  [!!!] OVERFIT: val_score = {r['overfit_pct']:.0f}% of train_score")
        print(f"     Formula memorized training data, fails out-of-sample.")
        print(f"     SUGGEST: discard, retrain with higher adversarial_trials.")
    elif r["overfit_pct"] > 0:
        print(f"\n  [OK] Generalization: val_score = {r['overfit_pct']:.0f}% of train")
    else:
        print(f"\n  [WARN] Train return negative, overfit check skipped")

    # Operator composition
    print(f"\n  [Ops]")
    for name, count in sorted(r["complexity"]["composition"].items(), key=lambda x: -x[1]):
        bar = "#" * min(count, 20)
        print(f"     {name:<8} {count:>2}x  {bar}")


# ─── CLI ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaGPT 公式样本外审计工具")
    parser.add_argument("inst_id", nargs="?", default="BTC-USDT", help="品种 (BTC-USDT)")
    parser.add_argument("bar", nargs="?", default="1H", help="K线周期")
    parser.add_argument("--formula", "-f", help="公式 JSON 路径 (默认 output/{INST}_{BAR}_formula.json)")
    parser.add_argument("--adversarial", "-a", type=int, default=0, help="对抗噪声试验次数")
    parser.add_argument("--noise", type=float, default=0.02, help="噪声强度")
    parser.add_argument("--split", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--all", action="store_true", help="评估 output/ 下所有公式")

    args = parser.parse_args()

    if args.all:
        import glob
        files = sorted(glob.glob("output/*_formula.json"))
        if not files:
            print("No formula files found in output/")
            sys.exit(1)
        for fp in files:
            with open(fp) as f:
                data = json.load(f)
            formula = data if isinstance(data, list) else data.get("formula", data.get("best_formula"))
            inst = data.get("inst_id", args.inst_id) if isinstance(data, dict) else args.inst_id
            bar = data.get("bar", args.bar) if isinstance(data, dict) else args.bar
            if isinstance(formula, list):
                r = evaluate_formula(formula, inst, bar, args.split, args.adversarial, args.noise)
                print_report(r)
    else:
        if args.formula:
            fp = args.formula
        else:
            prefix = f"{args.inst_id.replace('-','')}_{args.bar}"
            fp = f"output/{prefix}_formula.json"

        if not os.path.exists(fp):
            print(f"Formula file not found: {fp}")
            sys.exit(1)

        with open(fp) as f:
            data = json.load(f)
        formula = data if isinstance(data, list) else data.get("formula", data.get("best_formula"))

        if not isinstance(formula, list):
            print(f"Invalid formula in {fp}")
            sys.exit(1)

        r = evaluate_formula(formula, args.inst_id, args.bar, args.split, args.adversarial, args.noise)
        print_report(r)
