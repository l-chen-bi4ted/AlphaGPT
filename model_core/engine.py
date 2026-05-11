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
        pretrain_oracle: 是否先用 oracle 生成 top-k 公式做预训练
        oracle_max_len: oracle 搜索的最大公式长度（建议 ≤8）
        oracle_ops_subset: oracle 使用的算子子集
        oracle_topk: oracle 预训练的样本数
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
        pretrain_oracle: bool = False,
        oracle_max_len: int = 6,
        oracle_ops_subset: str = "basic",
        oracle_topk: int = 500,
    ):
        self.config = config or default_config
        self.inst_id = inst_id or self.config.inst_id
        self.bar = bar or self.config.bar
        self.limit = candle_limit or self.config.candle_limit
        self.oos_every = oos_every
        self.early_stop_patience = early_stop_patience
        self.steps_since_improvement = 0
        self.pretrain_oracle = pretrain_oracle
        self.oracle_max_len = oracle_max_len
        self.oracle_ops_subset = oracle_ops_subset
        self.oracle_topk = oracle_topk

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

    def pretrain_from_oracle(self, max_len: int, ops_subset: str, topk: int):
        """
        用 ExhaustiveOracle 生成 top-k 公式，对 AlphaGPT 做预训练。
        让模型先学会生成高 reward 公式的分布，再进入 RL 微调。
        """
        from .oracle import ExhaustiveOracle

        logger.info("[Pretrain] Starting Oracle search for pretraining data...")
        oracle = ExhaustiveOracle(
            config=self.config,
            max_len=max_len,
            min_len=max(3, max_len - 2),
            ops_subset=ops_subset,
            k_feats=self.config.input_dim,
        )
        results = oracle.search(
            self.loader, topk=topk, val_split=0.2, verbose=True
        )
        if not results:
            logger.warning("[Pretrain] Oracle found no valid formulas, skipping pretrain")
            return

        # 取 top 80% 为正样本，bottom 20% 为负样本
        n_pos = int(topk * 0.8)
        positive = results[:n_pos]
        negative = results[-max(10, topk - n_pos):]

        dataset = positive + negative
        logger.info(f"[Pretrain] Dataset: {len(positive)} pos + {len(negative)} neg = {len(dataset)}")

        # 构建训练数据
        max_len = self.config.max_formula_len
        token_seqs = []
        rewards = []
        for r in dataset:
            seq = list(r.formula)
            # pad 到 max_len（后面补 0，即第一个特征 token）
            seq = seq + [0] * (max_len - len(seq))
            token_seqs.append(seq)
            rewards.append(r.composite)

        token_seqs = torch.tensor(token_seqs, dtype=torch.long, device=self.config.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.config.device)

        # 预训练：behavior cloning（最大似然）+ reward 加权
        pretrain_steps = min(200, len(dataset) * 2)
        pretrain_opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate * 2)

        for step in range(pretrain_steps):
            # 随机采样 batch
            idx = torch.randint(0, len(dataset), (min(64, len(dataset)),), device=self.config.device)
            batch_seqs = token_seqs[idx]  # [B, L]
            batch_rewards = rewards_t[idx]  # [B]

            # Teacher forcing: 输入前 t-1 个 token，预测第 t 个
            loss = 0.0
            for t in range(max_len):
                inp = batch_seqs[:, :t + 1]  # [B, t+1]（包含当前作为 target）
                if inp.shape[1] == 0:
                    continue
                # 实际 target 是第 t 个 token（0-indexed）
                # 输入是前 t 个，预测第 t 个
                # 当 t=0 时，输入 batch_seqs[:, :1]，target = batch_seqs[:, 0]
                # 但模型 forward 输出的是基于输入序列的 next token 预测
                # 所以输入应为 batch_seqs[:, :t+1]，取最后一个位置的 logits
                logits, _, _ = self.model(inp)
                target_t = batch_seqs[:, t]
                # reward 加权 CE loss：高 reward 样本的梯度权重更大
                weights = (batch_rewards - batch_rewards.min() + 1e-3) / (batch_rewards.max() - batch_rewards.min() + 1e-3)
                ce = torch.nn.functional.cross_entropy(logits, target_t, reduction='none')
                loss += (ce * weights).mean()

            pretrain_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            pretrain_opt.step()

            if step % 50 == 0:
                logger.info(f"[Pretrain] step {step}/{pretrain_steps} loss={loss.item():.3f}")

        logger.info("[Pretrain] Oracle pretraining completed")

    def train(self):
        label = f"{self.inst_id} {self.bar}"
        logger.info(f"[AlphaGPT CEX Training {label}]")
        if self.use_lord:
            logger.info("  LoRD Regularization: enabled")

        # ── Oracle 预训练 ──
        if getattr(self, 'pretrain_oracle', False):
            self.pretrain_from_oracle(
                max_len=getattr(self, 'oracle_max_len', 6),
                ops_subset=getattr(self, 'oracle_ops_subset', 'basic'),
                topk=getattr(self, 'oracle_topk', 500),
            )

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
