#!/usr/bin/env python3
"""
快速超参扫描脚本 — 多组短训练 + 自动对抗筛选。

用法:
    python sweep.py                    # BTC 3组超参
    python sweep.py ETH-USDT           # 指定品种
    python sweep.py --steps 300        # 更快的快速扫描
"""

import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from model_core.config import ModelConfig

SWEEPS = [
    {"BATCH_SIZE": 4096, "LEARNING_RATE": 1e-3, "TRAIN_STEPS": 500, "desc": "baseline"},
    {"BATCH_SIZE": 8192, "LEARNING_RATE": 1e-3, "TRAIN_STEPS": 500, "desc": "large batch"},
    {"BATCH_SIZE": 4096, "LEARNING_RATE": 3e-3, "TRAIN_STEPS": 500, "desc": "high lr"},
]


def run_one(inst_id, bar, params):
    import torch
    from tqdm import tqdm
    from torch.distributions import Categorical
    from okx_data import OKXDataLoader
    from model_core.alphagpt import AlphaGPT
    from model_core.vm import StackVM
    from model_core.backtest import CEXBacktest

    # Override config
    ModelConfig.BATCH_SIZE = params["BATCH_SIZE"]
    ModelConfig.LEARNING_RATE = params["LEARNING_RATE"]
    ModelConfig.TRAIN_STEPS = params["TRAIN_STEPS"]

    desc = f"{inst_id} {params['desc']} (bs={params['BATCH_SIZE']} lr={params['LEARNING_RATE']} steps={params['TRAIN_STEPS']})"
    print(f"\n  [{desc}]")

    # Data
    loader = OKXDataLoader(inst_id, bar, 2000)
    loader.load_data()
    T = loader.feat_tensor.shape[-1]
    split = int(T * 0.8)
    train_feat = loader.feat_tensor[..., :split]
    train_target = loader.target_ret[..., :split]
    val_feat = loader.feat_tensor[..., split:]
    val_target = loader.target_ret[..., split:]

    # Model
    device = ModelConfig.DEVICE
    model = AlphaGPT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["LEARNING_RATE"])
    vm = StackVM()
    bt = CEXBacktest(adversarial_trials=0, multi_dim=True)

    best_formula = None
    best_train_score = -float("inf")

    pbar = tqdm(range(params["TRAIN_STEPS"]), desc=desc[:40], leave=False)
    for step in pbar:
        bs = params["BATCH_SIZE"]
        inp = torch.zeros((bs, 1), dtype=torch.long, device=device)
        log_probs, tokens_list = [], []

        for _ in range(ModelConfig.MAX_FORMULA_LEN):
            logits, _, _ = model(inp)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            tokens_list.append(action)
            inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

        seqs = torch.stack(tokens_list, dim=1)
        rewards = torch.zeros(bs, device=device)

        for i in range(bs):
            formula = seqs[i].tolist()
            res = vm.execute(formula, train_feat)
            if res is None or res.std() < 1e-4:
                rewards[i] = -5.0
                continue
            score, _ = bt.evaluate(res, loader.raw_data_cache, train_target)
            n_unique = len(set(formula))
            rewards[i] = score - max(0, n_unique - 5) * 0.02
            if score.item() > best_train_score:
                best_train_score = score.item()
                best_formula = formula

        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        loss = sum(-lp * adv for lp in log_probs).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    # ── 对抗筛选 ──
    if best_formula is None:
        return None

    res_train = vm.execute(best_formula, train_feat)
    res_val = vm.execute(best_formula, val_feat)

    bt_adv = CEXBacktest(adversarial_trials=3, noise_std=0.02, multi_dim=True)
    train_score, _ = bt_adv.evaluate(res_train, loader.raw_data_cache, train_target)
    val_score, val_ret = bt_adv.evaluate(res_val, loader.raw_data_cache, val_target)

    return {
        "inst_id": inst_id, "bar": bar,
        "params": params,
        "formula": best_formula,
        "train_score": train_score.item(),
        "val_score": val_score.item(),
        "val_return": val_ret,
        "n_unique": len(set(best_formula)),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("inst_id", nargs="?", default="BTC-USDT")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    for s in SWEEPS:
        s["TRAIN_STEPS"] = args.steps

    results = []
    for params in SWEEPS:
        r = run_one(args.inst_id, "1H", params)
        if r:
            results.append(r)
            print(f"  => train={r['train_score']:.2f} val={r['val_score']:.2f} ret={r['val_return']:.2%}")

    # Sort by val_score
    results.sort(key=lambda x: x["val_score"], reverse=True)

    print(f"\n{'='*60}")
    print(f"  SWEEP RESULTS: {args.inst_id}")
    print(f"{'='*60}")
    print(f"  {'Desc':<30} {'Train':>8} {'Val':>8} {'Ret':>8} {'Unique':>6}")
    print(f"  {'-'*50}")
    for r in results:
        if r["val_score"] > 0:
            marker = "[KEEP]"
        else:
            marker = "[DROP]"
        print(f"  {r['params']['desc']:<30} {r['train_score']:>8.2f} {r['val_score']:>8.2f} {r['val_return']:>7.2%} {r['n_unique']:>6}  {marker}")

    # Save best
    best = results[0] if results else None
    if best and best["val_score"] > 0:
        os.makedirs("output", exist_ok=True)
        name = args.inst_id.replace("-", "")
        fp = f"output/{name}_1H_sweep_best.json"
        with open(fp, "w") as f:
            json.dump({"inst_id": best["inst_id"], "bar": best["bar"],
                       "train_score": best["train_score"], "val_score": best["val_score"],
                       "formula": best["formula"]}, f, indent=2)
        print(f"\n  Best saved: {fp}")
    else:
        print(f"\n  No formula passed adversarial filter. Try more sweeps or different data.")
