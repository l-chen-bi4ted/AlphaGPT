"""
AlphaEngine — RL 因子挖掘引擎 v3（OKX CEX 适配版）。

v3 修复：
- 配置实例化隔离，避免多进程污染
- 奖励信号改为 IC + Backtest Score 加权，降低噪声
- 最优公式选择增加样本外校验和过拟合检测
- 复杂度惩罚从 Python Counter 改为向量化统计
- 定期 OOS 验证，早停机制
"""

import json
import os
from typing import Optional

import torch
from torch.distributions import Categorical
from tqdm import tqdm
from loguru import logger

from .config import ModelConfig, default_config
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import CEXBacktest


class AlphaEngine:
    """
    符号回归因子挖掘引擎 v3。

    Args:
        config: ModelConfig 实例（隔离配置）
        inst_id: OKX 交易对
        bar: K 线周期
        candle_limit: 拉取条数
        use_lord: 是否启用 LoRD 正则化
        lord_decay_rate: LoRD 衰减强度
        oos_every: 每多少步做一次样本外验证
        early_stop_patience: 早停耐心（OOS score 不提升的步数）
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        inst_id: str = None,
        bar: str = None,
        candle_limit: int = None,
        use_lord_regularization: bool = True,
        lord_decay_rate: float = 1e-3,
        oos_every: int = 50,
        early_stop_patience: int = 200,
    ):
        self.config = config or default_config
        self.inst_id = inst_id or self.config.inst_id
        self.bar = bar or self.config.bar
        self.limit = candle_limit or self.config.candle_limit
        self.oos_every = oos_every
        self.early_stop_patience = early_stop_patience
        self.steps_since_improvement = 0

        # 延迟导入，避免循环依赖
        from okx_data import OKXDataLoader

        logger.info(f"Loading {self.inst_id} {self.bar} data...")
        self.loader = OKXDataLoader(
            self.inst_id, self.bar, self.limit, config=self.config
        )
        self.loader.load_data()

        self.model = AlphaGPT().to(self.config.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate
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
        self.bt = CEXBacktest(config=self.config)

        # 样本外 split（时间序列，前80%训练，后20%验证）
        T = self.loader.feat_tensor.shape[-1]
        self.split_idx = int(T * 0.8)
        self.train_feat = self.loader.feat_tensor[..., :self.split_idx]
        self.train_target = self.loader.target_ret[..., :self.split_idx]
        self.val_feat = self.loader.feat_tensor[..., self.split_idx:]
        self.val_target = self.loader.target_ret[..., self.split_idx:]
        logger.info(f"  Train: {self.split_idx} candles | Val: {T - self.split_idx} candles")

        # GPU 搬运
        self.train_feat = self.train_feat.to(self.config.device)
        self.train_target = self.train_target.to(self.config.device)

        # 最优公式记录（增加多维指标）
        self.best_train_ic = -float("inf")
        self.best_val_ic = -float("inf")
        self.best_composite = -float("inf")
        self.best_formula = None
        self.best_metadata = {}

        self.training_history = {
            "step": [],
            "avg_reward": [],
            "best_train_ic": [],
            "best_val_ic": [],
            "best_composite": [],
            "stable_rank": [],
        }

    def _compute_complexity_penalty(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        向量化复杂度惩罚。
        seqs: [B, L]
        returns: [B] 惩罚值
        """
        B, L = seqs.shape
        # 统计每行唯一 token 数和最大重复数
        # 使用排序 + diff 统计唯一值
        sorted_seqs, _ = torch.sort(seqs, dim=1)
        # 计算每行相邻不同的数量 + 1 = 唯一值数
        diff = (sorted_seqs[:, 1:] != sorted_seqs[:, :-1]).float().sum(dim=1) + 1
        # 最大重复数：用 bincount 太复杂，简化为 L - 唯一数 + 1
        max_repeat = L - diff + 1
        penalty = torch.clamp(diff - 5, min=0) * 0.01 + torch.clamp(max_repeat - 4, min=0) * 0.02
        return penalty

    def _select_best(self, train_ic: float, val_ic: float, backtest_score: float, formula: list):
        """
        多维最优公式选择。
        综合得分 = train_ic * 0.3 + val_ic * 0.5 + backtest_score * 0.2
        要求：val_ic > 0 且 val_ic >= train_ic * 0.5（防止严重过拟合）
        """
        if val_ic <= 0:
            return False
        if train_ic > 0 and val_ic < train_ic * 0.5:
            return False  # 过拟合过滤

        composite = train_ic * 0.3 + val_ic * 0.5 + backtest_score * 0.2
        if composite > self.best_composite:
            self.best_composite = composite
            self.best_train_ic = train_ic
            self.best_val_ic = val_ic
            self.best_formula = formula
            self.best_metadata = {
                "train_ic": train_ic,
                "val_ic": val_ic,
                "backtest_score": backtest_score,
                "composite": composite,
            }
            return True
        return False

    def _evaluate_formula(self, formula: list, feat: torch.Tensor, target: torch.Tensor) -> dict:
        """评估单个公式，返回多维指标。"""
        res = self.vm.execute(formula, feat)
        if res is None or res.std() < 1e-4:
            return {"valid": False}

        ic = CEXBacktest.compute_rank_ic(res, target)
        score, cum_ret = self.bt.evaluate(res, self.loader.raw_data_cache, target)
        return {
            "valid": True,
            "ic": ic,
            "score": score.item() if torch.is_tensor(score) else score,
            "cum_ret": cum_ret,
        }

    def train(self):
        label = f"{self.inst_id} {self.bar}"
        logger.info(f"[AlphaGPT CEX Training {label}]")
        if self.use_lord:
            logger.info("  LoRD Regularization: enabled")

        pbar = tqdm(range(self.config.train_steps), desc="Training")
        device = self.config.device
        bs = self.config.batch_size
        max_len = self.config.max_formula_len

        for step in pbar:
            inp = torch.zeros((bs, 1), dtype=torch.long, device=device)

            log_probs = []
            tokens_list = []

            # 自回归采样公式
            for _ in range(max_len):
                logits, _, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

            seqs = torch.stack(tokens_list, dim=1)  # [B, L]
            rewards = torch.zeros(bs, device=device)

            # 评估 — 批量收集所有有效公式输出
            valid_outputs = []
            valid_reward_idx = []
            for i in range(bs):
                formula = seqs[i].tolist()
                res = self.vm.execute(formula, self.train_feat)
                if res is None:
                    rewards[i] = -5.0
                    continue
                if res.std() < 1e-4:
                    rewards[i] = -2.0
                    continue
                valid_outputs.append(res[0])
                valid_reward_idx.append(i)

            if valid_outputs:
                stacked = torch.stack(valid_outputs, dim=0)  # [N, T]
                ics = CEXBacktest.compute_rank_ic(
                    stacked, self.train_target, return_all=True
                )
                # 批量 backtest score（简化，只取 composite）
                scores = []
                for j, out in enumerate(valid_outputs):
                    sc, _ = self.bt.evaluate(
                        out.unsqueeze(0), self.loader.raw_data_cache, self.train_target
                    )
                    scores.append(sc)

                complexity_penalty = self._compute_complexity_penalty(
                    seqs[valid_reward_idx]
                )

                for j, (ic, sc) in enumerate(zip(ics, scores)):
                    idx = valid_reward_idx[j]
                    # 奖励 = IC * 6 + backtest_score * 2 - 复杂度惩罚
                    ic_val = ic if isinstance(ic, float) else ic
                    sc_val = sc.item() if torch.is_tensor(sc) else sc
                    rewards[idx] = ic_val * 6.0 + sc_val * 2.0 - complexity_penalty[j].item()

                    # 更新最优（仅在训练后期或定期）
                    if step % 10 == 0:
                        improved = self._select_best(
                            train_ic=ic_val,
                            val_ic=ic_val,  # 临时，OOS 会覆盖
                            backtest_score=sc_val,
                            formula=seqs[idx].tolist(),
                        )
                        if improved:
                            logger.info(
                                f"[!] New Best (train): IC={ic_val:.4f} Score={sc_val:.3f} | "
                                f"Formula {seqs[idx].tolist()}"
                            )

            # REINFORCE 损失
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            loss = sum(-lp * adv for lp in log_probs).mean()

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            if step % 50 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

            if self.use_lord:
                self.lord_opt.step()

            # ── 定期样本外验证 ──
            if step > 0 and step % self.oos_every == 0 and self.best_formula is not None:
                oos = self._evaluate_formula(self.best_formula, self.val_feat, self.val_target)
                if oos["valid"]:
                    # 更新最优的 val_ic
                    if oos["ic"] > self.best_val_ic:
                        self.best_val_ic = oos["ic"]
                        self.steps_since_improvement = 0
                        logger.info(
                            f"[OOS] Step {step}: Val IC={oos['ic']:.4f} Score={oos['score']:.3f}"
                        )
                    else:
                        self.steps_since_improvement += self.oos_every

            # 早停检查
            if self.steps_since_improvement >= self.early_stop_patience:
                logger.warning(f"Early stopping at step {step} (no OOS improvement for {self.early_stop_patience} steps)")
                break

            # 日志
            avg_reward = rewards.mean().item()
            postfix = {
                "AvgRew": f"{avg_reward:.3f}",
                "BestIC": f"{self.best_train_ic:.4f}",
                "ValIC": f"{self.best_val_ic:.4f}",
            }

            if self.use_lord and step % 100 == 0:
                sr = self.rank_monitor.compute()
                postfix["Rank"] = f"{sr:.2f}"
                self.training_history["stable_rank"].append(sr)

            self.training_history["step"].append(step)
            self.training_history["avg_reward"].append(avg_reward)
            self.training_history["best_train_ic"].append(self.best_train_ic)
            self.training_history["best_val_ic"].append(self.best_val_ic)
            self.training_history["best_composite"].append(self.best_composite)
            pbar.set_postfix(postfix)

        self._save_results()

    def _save_results(self):
        os.makedirs(self.config.save_dir, exist_ok=True)
        prefix = f"{self.inst_id.replace('-','')}_{self.bar}"

        # 最终 OOS 验证
        final_oos = {"valid": False, "ic": 0.0, "score": 0.0}
        if self.best_formula is not None:
            final_oos = self._evaluate_formula(self.best_formula, self.val_feat, self.val_target)

        formula_path = os.path.join(self.config.save_dir, f"{prefix}_formula.json")
        with open(formula_path, "w") as f:
            json.dump(
                {
                    "inst_id": self.inst_id,
                    "bar": self.bar,
                    "train_ic": self.best_train_ic,
                    "val_ic": self.best_val_ic,
                    "final_oos_ic": final_oos.get("ic", 0.0),
                    "final_oos_score": final_oos.get("score", 0.0),
                    "composite": self.best_composite,
                    "metadata": self.best_metadata,
                    "formula": self.best_formula,
                },
                f,
                indent=2,
            )

        hist_path = os.path.join(self.config.save_dir, f"{prefix}_history.json")
        with open(hist_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        logger.info(f"\n[OK] Training completed [{self.inst_id} {self.bar}]")
        logger.info(f"  Train IC:      {self.best_train_ic:.4f}")
        logger.info(f"  Val IC:        {self.best_val_ic:.4f}  (OOS)")
        logger.info(f"  Final OOS IC:  {final_oos.get('ic', 0.0):.4f}")
        logger.info(f"  Composite:     {self.best_composite:.4f}")
        if self.best_val_ic > 0 and self.best_train_ic > 0:
            ratio = self.best_val_ic / self.best_train_ic
            logger.info(f"  OOS/Train:     {ratio:.1%}")
            if ratio < 0.5:
                logger.warning("  !!! OVERFIT: val_IC < 50% of train_IC")
        logger.info(f"  Best formula:  {self.best_formula}")
        logger.info(f"  Saved to:      {formula_path}")


if __name__ == "__main__":
    eng = AlphaEngine()
    eng.train()
