#!/usr/bin/env python3
"""E2E test: data loading → training → formula output."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== 1. Data Loading ===")
from okx_data import OKXDataLoader
loader = OKXDataLoader("BTC-USDT", "1H", 200)
loader.load_data()
print(f"feat_tensor shape: {loader.feat_tensor.shape}")
print(f"target_ret shape: {loader.target_ret.shape}")
print()

print("=== 2. Model Init ===")
from model_core.config import ModelConfig
from model_core.alphagpt import AlphaGPT
from model_core.vm import StackVM
from model_core.backtest import CEXBacktest

import torch

# Override for quick test
ModelConfig.TRAIN_STEPS = 50
ModelConfig.BATCH_SIZE = 256

model = AlphaGPT().to(ModelConfig.DEVICE)
print(f"Vocab size: {model.vocab_size} (expect 22 = 6 features + 16 ops)")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
print()

print("=== 3. VM + Backtest Sanity ===")
vm = StackVM()
bt = CEXBacktest()

# Simple formula: just feature 0 (RET)
formula = [0]
res = vm.execute(formula, loader.feat_tensor)
if res is not None:
    score, ret = bt.evaluate(res, loader.raw_data_cache, loader.target_ret)
    print(f"RET-only: score={score:.3f}, ret={ret:.4%}")
print()

print("=== 4. Training (50 steps) ===")
from torch.distributions import Categorical

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
device = ModelConfig.DEVICE
best_score = -float("inf")
best_formula = None

for step in range(ModelConfig.TRAIN_STEPS):
    bs = ModelConfig.BATCH_SIZE
    inp = torch.zeros((bs, 1), dtype=torch.long, device=device)

    log_probs = []
    tokens_list = []

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
        res = vm.execute(formula, loader.feat_tensor)
        if res is None:
            rewards[i] = -5.0
            continue
        if res.std() < 1e-4:
            rewards[i] = -2.0
            continue
        score, ret_val = bt.evaluate(res, loader.raw_data_cache, loader.target_ret)
        rewards[i] = score
        if score.item() > best_score:
            best_score = score.item()
            best_formula = formula

    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
    loss = sum(-lp * adv for lp in log_probs).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 10 == 0:
        avg_reward = rewards.mean().item()
        print(f"  Step {step:3d}: avg_reward={avg_reward:.3f}, best={best_score:.3f}")

print()
print(f"=== 5. Results ===")
print(f"Best score: {best_score:.4f}")
print(f"Best formula: {best_formula}")
print()
print("E2E TEST PASSED" if best_score > -float("inf") else "E2E TEST FAILED")
