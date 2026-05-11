#!/usr/bin/env python3
"""E2E test v3: data loading → training → formula output."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_core.config import ModelConfig
from model_core.alphagpt import AlphaGPT
from model_core.vm import StackVM
from model_core.backtest import CEXBacktest
import torch
from torch.distributions import Categorical

print("=== 1. Data Loading ===")
from okx_data import OKXDataLoader
config = ModelConfig(train_steps=50, batch_size=256)
loader = OKXDataLoader("BTC-USDT", "1H", 200, config=config)
loader.load_data()
print(f"feat_tensor shape: {loader.feat_tensor.shape}")
print(f"target_ret shape: {loader.target_ret.shape}")
print()

print("=== 2. Model Init ===")
model = AlphaGPT(config=config).to(config.device)
print(f"Vocab size: {model.vocab_size} (expect 22 = 6 features + 16 ops)")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
print()

print("=== 3. VM + Backtest Sanity ===")
vm = StackVM()
bt = CEXBacktest(config=config)

# Simple formula: just feature 0 (RET)
formula = [0]
res = vm.execute(formula, loader.feat_tensor)
if res is not None:
    score, ret = bt.evaluate(res, loader.raw_data_cache, loader.target_ret)
    print(f"RET-only: score={score:.3f}, ret={ret:.4%}")
print()

print(f"=== 4. Training ({config.train_steps} steps) ===")
opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
device = config.device
best_score = -float("inf")
best_formula = None

for step in range(config.train_steps):
    bs = config.batch_size
    inp = torch.zeros((bs, 1), dtype=torch.long, device=device)

    log_probs = []
    tokens_list = []

    for _ in range(config.max_formula_len):
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
