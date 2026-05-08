"""
AlphaEngine — RL 因子挖掘引擎（OKX CEX 适配版）。

训练循环：
  1. OKXDataLoader 拉取 K 线 → 特征张量
  2. AlphaGPT 采样公式 token 序列 → StackVM 执行
  3. CEXBacktest 评估 → REINFORCE 梯度更新
  4. Newton-Schulz LoRD 正则化
"""

import json
import os

import torch
from torch.distributions import Categorical
from tqdm import tqdm

from .config import ModelConfig
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import CEXBacktest


class AlphaEngine:
    """
    符号回归因子挖掘引擎。

    Args:
        inst_id: OKX 交易对（默认 BTC-USDT）
        bar: K 线周期（默认 1H）
        candle_limit: 拉取条数
        use_lord: 是否启用 LoRD 正则化
        lord_decay_rate: LoRD 衰减强度
    """

    def __init__(
        self,
        inst_id: str = None,
        bar: str = None,
        candle_limit: int = None,
        use_lord_regularization: bool = True,
        lord_decay_rate: float = 1e-3,
    ):
        # 延迟导入，避免循环依赖
        from okx_data import OKXDataLoader

        self.inst_id = inst_id or ModelConfig.INST_ID
        self.bar = bar or ModelConfig.BAR
        self.limit = candle_limit or ModelConfig.CANDLE_LIMIT

        print(f"Loading {self.inst_id} {self.bar} data...")
        self.loader = OKXDataLoader(self.inst_id, self.bar, self.limit)
        self.loader.load_data()

        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=ModelConfig.LEARNING_RATE
        )

        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"],
            )
            self.rank_monitor = StableRankMonitor(
                self.model, target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = StackVM()
        self.bt = CEXBacktest()

        self.best_score = -float("inf")
        self.best_formula = None
        self.training_history = {
            "step": [],
            "avg_reward": [],
            "best_score": [],
            "stable_rank": [],
        }

    def train(self):
        label = f"{self.inst_id} {self.bar}"
        print(f"[AlphaGPT CEX Training {label}]")
        if self.use_lord:
            print("   LoRD Regularization: enabled")

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        device = ModelConfig.DEVICE

        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=device)

            log_probs = []
            tokens_list = []

            # 自回归采样公式
            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

            seqs = torch.stack(tokens_list, dim=1)  # [B, L]
            rewards = torch.zeros(bs, device=device)

            # 评估每个采样公式
            for i in range(bs):
                formula = seqs[i].tolist()
                res = self.vm.execute(formula, self.loader.feat_tensor)
                if res is None:
                    rewards[i] = -5.0
                    continue
                if res.std() < 1e-4:
                    rewards[i] = -2.0
                    continue
                score, ret_val = self.bt.evaluate(
                    res, self.loader.raw_data_cache, self.loader.target_ret
                )
                rewards[i] = score

                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula
                    tqdm.write(
                        f"[!] New Best: Score {score:.2f} | Ret {ret_val:.2%} | "
                        f"Formula {formula}"
                    )

            # REINFORCE 损失
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            loss = sum(-lp * adv for lp in log_probs).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if self.use_lord:
                self.lord_opt.step()

            # 日志
            avg_reward = rewards.mean().item()
            postfix = {
                "AvgRew": f"{avg_reward:.3f}",
                "Best": f"{self.best_score:.3f}",
            }

            if self.use_lord and step % 100 == 0:
                sr = self.rank_monitor.compute()
                postfix["Rank"] = f"{sr:.2f}"
                self.training_history["stable_rank"].append(sr)

            self.training_history["step"].append(step)
            self.training_history["avg_reward"].append(avg_reward)
            self.training_history["best_score"].append(self.best_score)
            pbar.set_postfix(postfix)

        self._save_results()

    def _save_results(self):
        os.makedirs(ModelConfig.SAVE_DIR, exist_ok=True)

        prefix = f"{self.inst_id.replace('-','')}_{self.bar}"

        # 最优公式
        formula_path = os.path.join(ModelConfig.SAVE_DIR, f"{prefix}_formula.json")
        with open(formula_path, "w") as f:
            json.dump(
                {
                    "inst_id": self.inst_id,
                    "bar": self.bar,
                    "score": self.best_score,
                    "formula": self.best_formula,
                },
                f,
                indent=2,
            )

        # 训练历史
        hist_path = os.path.join(ModelConfig.SAVE_DIR, f"{prefix}_history.json")
        with open(hist_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        print(f"\n✓ Training completed [{self.inst_id} {self.bar}]")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")
        print(f"  Saved to: {formula_path}")


if __name__ == "__main__":
    eng = AlphaEngine()
    eng.train()
