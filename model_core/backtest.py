"""
CEX 回测引擎 v2 — 对抗评估 + 多维 fitness。

v2 新增：
- 对抗噪声评估：对因子序列注入微小扰动，取 worst-case score
- 多维 fitness：Sharpe / Sortino / 胜率 / 盈亏比 综合打分
- 事后最优参考：计算 regret（与完美策略的差距）

设计哲学（来自 no_JIT / HJI 微分博弈）：
  市场是对手方，好策略不是"历史赚了多少"，而是"最坏情况下仍不亏"。
"""

import torch
import numpy as np
from .config import ModelConfig


class CEXBacktest:
    """
    CEX 现货回测 v2。

    Args:
        fee_rate: 单边手续费率（默认 0.001 = 0.1%）
        trade_size: 每笔名义交易金额（USD）
        signal_threshold: sigmoid 信号 > 此值才入场
        min_trades: 最少交易次数，低于此则判定为无效公式
        adversarial_trials: 对抗噪声试验次数（0=关闭）
        noise_std: 对抗噪声标准差（相对因子 std 的比例）
        multi_dim: 是否启用多维 fitness（Sharpe+Sortino+胜率+盈亏比）
    """

    def __init__(
        self,
        fee_rate: float = None,
        trade_size: float = None,
        signal_threshold: float = None,
        min_trades: int = 5,
        adversarial_trials: int = 0,
        noise_std: float = 0.01,
        multi_dim: bool = True,
    ):
        self.fee_rate = fee_rate if fee_rate is not None else ModelConfig.BASE_FEE
        self.trade_size = trade_size if trade_size is not None else ModelConfig.TRADE_SIZE_USD
        self.threshold = signal_threshold if signal_threshold is not None else ModelConfig.SIGNAL_THRESHOLD
        self.min_trades = min_trades
        self.adversarial_trials = adversarial_trials
        self.noise_std = noise_std
        self.multi_dim = multi_dim

    # ─── 单次评估核心 ──────────────────────────────────
    def _eval_once(
        self,
        factors: torch.Tensor,
        target_ret: torch.Tensor,
    ) -> dict:
        """对因子序列做一次评估，返回详细指标字典。"""
        if factors.dim() == 1:
            factors = factors.unsqueeze(0)
        B, T = factors.shape

        signal = torch.sigmoid(factors)  # [B, T]
        position = (signal > self.threshold).float()

        # 交易成本
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * self.fee_rate

        # PnL
        gross_pnl = position * target_ret  # [B, T]
        net_pnl = gross_pnl - tx_cost

        # 每行独立评估，取中位数
        cum_ret = net_pnl.sum(dim=1)  # [B]
        activity = position.sum(dim=1)

        # 过滤无效公式
        valid = activity >= self.min_trades

        if valid.sum() == 0:
            return {
                "score": torch.tensor(-10.0, device=factors.device),
                "cum_ret": 0.0,
                "sharpe": 0.0,
                "win_rate": 0.0,
                "max_dd": 1.0,
                "n_trades": 0,
                "valid": False,
            }

        # 只评估有效公式
        valid_cum = cum_ret[valid]
        valid_pnl = net_pnl[valid]

        median_ret = torch.median(valid_cum)

        # 多维综合打分（昂贵指标仅在需要时计算）
        sharpe = 0.0
        sortino = 0.0
        win_rate = 0.0
        max_dd = 0.0

        if self.multi_dim:
            # Sharpe（年化近似：假设 1H K 线，T 约 200）
            mean_pnl = valid_pnl.mean(dim=1)
            std_pnl = valid_pnl.std(dim=1) + 1e-8
            sharpe = (mean_pnl / std_pnl).median().item() * np.sqrt(365 * 24)

            # Sortino（只惩罚下行）
            downside = torch.clamp(valid_pnl, max=0)
            down_std = downside.std(dim=1) + 1e-8
            sortino = (mean_pnl / down_std).median().item() * np.sqrt(365 * 24)

            # 胜率
            wins = (valid_pnl > 0).float().sum(dim=1)
            total_trades = activity[valid].float()
            win_rate = (wins / (total_trades + 1e-8)).median().item()

            # 最大回撤
            cumsum = valid_pnl.cumsum(dim=1)
            running_max = cumsum.cummax(dim=1).values
            drawdown = running_max - cumsum
            max_dd = drawdown.max(dim=1).values.median().item()

            # 多维综合打分
            score = (
                np.clip(sharpe, -5, 5) * 0.3
                + np.clip(sortino, -5, 5) * 0.3
                + win_rate * 0.2
                + (1.0 - min(max_dd, 1.0)) * 0.2
            )

            # ── HJI Regret：与完美预知的差距 ──
            perfect_ret = self._perfect_foresight_return(target_ret)
            formula_ret = median_ret.item()
            # 只在完美策略能赚钱时才计算 regret
            opportunity = max(perfect_ret, 0)
            captured = max(formula_ret, 0)
            if opportunity > 0.01:
                regret = (opportunity - captured) / opportunity  # 0~1
                regret_penalty = regret * 0.5
                score -= regret_penalty
            elif formula_ret > 0 and perfect_ret < 0:
                # 完美策略亏钱但公式赚钱 → 加分
                score += 0.2

            # 交易太少惩罚
            median_activity = activity[valid].median().item()
            if median_activity < self.min_trades * 2:
                score -= 2.0

            # ── 奥卡姆剃刀：复杂度惩罚 ──
            # 无法从 factors 反推公式，此处留空但保留接口
            # 实际惩罚在 engine.py 的 rewards 计算中应用
        else:
            # 兼容旧版打分
            big_dd = (valid_pnl < -0.02).float().sum(dim=1)
            score = median_ret - big_dd.median() * 1.0

        return {
            "score": torch.tensor(score, device=factors.device, dtype=torch.float32),
            "cum_ret": median_ret.item(),
            "sharpe": sharpe,
            "sortino": sortino,
            "win_rate": win_rate,
            "max_dd": max_dd,
            "n_trades": activity[valid].median().item(),
            "perfect_ret": CEXBacktest._perfect_foresight_return(target_ret) if self.multi_dim else 0.0,
            "valid": True,
        }

    # ─── 完美预知策略（HJI regret 基准） ──────────────
    @staticmethod
    def _perfect_foresight_return(target_ret: torch.Tensor) -> float:
        """
        如果提前知道每一期的收益率，最优策略能赚多少。

        策略：下期涨则做多，跌则做空/空仓，扣手续费。
        这个值定义了 regret 的上界——任何实际策略都不可能超过它。
        """
        if target_ret.dim() == 2:
            target_ret = target_ret[0]  # [1, T] → [T]
        T = len(target_ret)
        ret = target_ret.cpu().numpy()

        # 完美策略：涨就持有，跌就空仓（费率扣在做多→空仓的换手上）
        # 实际上：每次都押对方向，只在方向翻转时扣一次手续费
        position = 1.0  # 起始做多
        total = 0.0

        for t in range(T - 1):
            total += position * ret[t + 1]
            # 方向翻转（信号变化时扣费）
            new_pos = 1.0 if ret[t + 1] > 0 else 0.0
            if new_pos != position:
                total -= ModelConfig.BASE_FEE
            position = new_pos

        return float(total)

    # ─── Rank IC 计算（因子质量的核心指标）───────────
    @staticmethod
    def compute_rank_ic(
        factors: torch.Tensor,
        target_ret: torch.Tensor,
        lag: int = 0,
    ) -> float:
        """
        计算因子与未来收益的 Rank IC（Spearman 秩相关系数）。

        不依赖回测收益，只看因子排序和实际收益排序的相关性。
        IC > 0.03 有效，> 0.05 优秀，> 0.1 极强。

        Args:
            factors: [B, T] 或 [T] 因子时间序列
            target_ret: [1, T] 目标收益率（已对齐，t 时刻因子预测 t+1→t+2 收益）
            lag: 额外滞后（0=使用原始 target_ret，N=再延迟 N 期）

        Returns:
            median IC（跨 batch 的中位数）
        """
        if factors.dim() == 1:
            factors = factors.unsqueeze(0)
        B, T = factors.shape

        # 只取有效区间（目标收益已知的区间）
        valid_len = target_ret.shape[-1] - lag
        if valid_len <= 10:
            return 0.0

        target = target_ret[0, lag:lag + valid_len].cpu().numpy()
        ics = []

        for i in range(B):
            f = factors[i, :valid_len].cpu().numpy()

            # 跳过常数/无效输出
            f_std = np.nanstd(f)
            if f_std < 1e-6 or np.isnan(f_std):
                ics.append(0.0)
                continue

            # 去掉 target_ret 中的零值位置
            mask = np.abs(target) > 1e-9
            if mask.sum() < 10:
                ics.append(0.0)
                continue

            try:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(f[mask], target[mask])
            except (ImportError, ModuleNotFoundError):
                # 纯 numpy 后备：先转秩再算 Pearson
                from numpy import argsort
                def rankdata(a):
                    n = len(a)
                    ikey = argsort(a)
                    result = np.empty(n)
                    result[ikey] = np.arange(1, n + 1)
                    # 处理并列：取平均秩
                    for val in np.unique(a):
                        idx = np.where(a == val)[0]
                        if len(idx) > 1:
                            result[idx] = result[idx].mean()
                    return result
                f_rank = rankdata(f[mask])
                t_rank = rankdata(target[mask])
                ic = np.corrcoef(f_rank, t_rank)[0, 1]
            except Exception:
                ic = 0.0

            ics.append(0.0 if np.isnan(ic) else ic)

        return float(np.median(ics))

    # ─── 主评估入口 ──────────────────────────────────
    def evaluate(
        self,
        factors: torch.Tensor,
        raw_data: dict,
        target_ret: torch.Tensor,
    ) -> tuple:
        """
        评估因子公式的表现。支持对抗噪声模式。

        Returns:
            score: 综合 fitness（用于 RL reward）
            mean_return: 平均累计收益
        """
        # ── 对抗噪声 ──
        if self.adversarial_trials > 0:
            scores = []
            returns = []

            # 基准：无噪声
            base = self._eval_once(factors, target_ret)
            scores.append(base["score"])
            returns.append(base["cum_ret"])

            # 噪声试验
            factor_std = factors.std() + 1e-8
            for _ in range(self.adversarial_trials - 1):
                noise = torch.randn_like(factors) * factor_std * self.noise_std
                noisy_factors = factors + noise
                result = self._eval_once(noisy_factors, target_ret)
                scores.append(result["score"])
                returns.append(result["cum_ret"])

            # worst-case 作为最终 fitness（min-max 哲学）
            scores_t = torch.stack([s.to(factors.device) for s in scores])
            final_score = scores_t.min()  # 取最差值
            mean_return = sum(returns) / len(returns)

            return final_score, mean_return

        # ── 标准单次评估 ──
        result = self._eval_once(factors, target_ret)
        return result["score"], result["cum_ret"]
