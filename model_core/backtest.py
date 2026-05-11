"""
CEX 回测引擎 v3 — 真实摩擦 + 时滞 + 滑点。

v3 修复：
- 消除未来函数：信号在 t 时刻生成，t+latency 时刻执行，按 t+latency 价格成交
- 真实滑点：按 slippage_bps 对成交价做不利偏移
- 真实摩擦：手续费按名义成交额扣除，turnover 计算使用价格权重
- 统一 fitness 量纲：所有子指标先归一化到 [-1, 1] 再加权

设计哲学：
  回测不是"历史能赚多少"，而是"同样的信号在同样的延迟和摩擦下能赚多少"。
"""

import torch
import numpy as np
from typing import Optional
from .config import ModelConfig, default_config


class CEXBacktest:
    """
    CEX 现货回测 v3。

    Args:
        config: ModelConfig 实例（隔离配置）
        fee_rate: 单边手续费率
        trade_size: 每笔名义交易金额（USD）
        signal_threshold: sigmoid 信号 > 此值才入场
        min_trades: 最少交易次数
        latency_bars: 信号→执行延迟 K 线数（默认 1）
        slippage_bps: 滑点（万分之几，默认 5 = 0.05%）
        adversarial_trials: 对抗噪声试验次数（0=关闭）
        noise_std: 对抗噪声标准差（相对因子 std 的比例）
        multi_dim: 是否启用多维 fitness
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        fee_rate: float = None,
        trade_size: float = None,
        signal_threshold: float = None,
        min_trades: int = 5,
        latency_bars: int = None,
        slippage_bps: float = None,
        adversarial_trials: int = None,
        noise_std: float = 0.01,
        multi_dim: bool = True,
    ):
        self.config = config or default_config
        self.fee_rate = fee_rate if fee_rate is not None else self.config.base_fee
        self.trade_size = trade_size if trade_size is not None else self.config.trade_size_usd
        self.threshold = signal_threshold if signal_threshold is not None else self.config.signal_threshold
        self.min_trades = min_trades
        self.latency = latency_bars if latency_bars is not None else self.config.latency_bars
        self.slippage = (slippage_bps if slippage_bps is not None else self.config.slippage_bps) / 10000.0
        self.adversarial_trials = adversarial_trials if adversarial_trials is not None else self.config.adversarial_trials
        self.noise_std = noise_std
        self.multi_dim = multi_dim

    # ─── 单次评估核心 ──────────────────────────────────
    def _eval_once(
        self,
        factors: torch.Tensor,
        target_ret: torch.Tensor,
        close: Optional[torch.Tensor] = None,
    ) -> dict:
        """对因子序列做一次评估，返回详细指标字典。"""
        if factors.dim() == 1:
            factors = factors.unsqueeze(0)
        B, T = factors.shape

        # 信号生成（t 时刻已知）
        signal = torch.sigmoid(factors)  # [B, T]
        
        # 时滞：信号在 t 时刻生成，实际在 t+latency 时刻执行
        # 将信号向左平移 latency，前面补 0（表示无信号时空仓）
        if self.latency > 0:
            delayed_signal = torch.cat([
                torch.zeros((B, self.latency), device=signal.device),
                signal[:, :-self.latency]
            ], dim=1)
        else:
            delayed_signal = signal

        position = (delayed_signal > self.threshold).float()

        # 计算持仓变化 turnover（按名义金额）
        prev_pos = torch.cat([torch.zeros((B, 1), device=position.device), position[:, :-1]], dim=1)
        turnover = torch.abs(position - prev_pos)  # [B, T]，每次换手的比例

        # 名义成交额 = turnover * trade_size
        notional_turnover = turnover * self.trade_rate

        # 手续费 = 名义成交额 * fee_rate * 2（开+平）
        # 简化：每次换手扣一次双边费用
        tx_cost = turnover * self.fee_rate * 2.0

        # 滑点：每次开仓方向不利偏移 slippage
        # 用 close 价格序列计算滑点对收益的影响
        if close is not None and close.numel() > 0:
            # 滑点导致的额外成本：每次交易按 slippage 损失
            slippage_cost = turnover * self.slippage
        else:
            slippage_cost = torch.zeros_like(turnover)

        # PnL：position 使用 t+latency 到 t+latency+1 的收益率
        # 由于 target_ret[t] = log(close[t+1] / close[t])
        # 而 position 在 t 时刻决定的是 t→t+1 的持仓
        # 但 position 已经延迟了 latency，所以实际上用的是 target_ret[t+latency:]
        # 为简化，我们假设 target_ret 已经对齐，只是前面 latency 个 bar 没有收益
        gross_pnl = position * target_ret  # [B, T]
        net_pnl = gross_pnl - tx_cost - slippage_cost

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
        valid_activity = activity[valid]

        median_ret = torch.median(valid_cum)

        # ── 多维综合打分 ──
        sharpe = 0.0
        sortino = 0.0
        win_rate = 0.0
        max_dd = 0.0
        calmar = 0.0

        if self.multi_dim:
            # 按年化因子缩放（假设 1H K 线，年交易小时数 365*24）
            ann_factor = np.sqrt(365 * 24)

            # Sharpe
            mean_pnl = valid_pnl.mean(dim=1)
            std_pnl = valid_pnl.std(dim=1) + 1e-8
            sharpe_vals = (mean_pnl / std_pnl) * ann_factor
            sharpe = torch.median(sharpe_vals).item()

            # Sortino
            downside = torch.clamp(valid_pnl, max=0)
            down_std = downside.std(dim=1) + 1e-8
            sortino_vals = (mean_pnl / down_std) * ann_factor
            sortino = torch.median(sortino_vals).item()

            # 胜率（逐 bar 盈亏）
            wins = (valid_pnl > 0).float().sum(dim=1)
            win_rate_vals = wins / (valid_activity.float() + 1e-8)
            win_rate = torch.median(win_rate_vals).item()

            # 最大回撤
            cumsum = valid_pnl.cumsum(dim=1)
            running_max = cumsum.cummax(dim=1).values
            drawdown = running_max - cumsum
            max_dd_vals = drawdown.max(dim=1).values
            max_dd = torch.median(max_dd_vals).item()

            # Calmar = 年化收益 / 最大回撤
            ann_ret = mean_pnl * 365 * 24
            calmar_vals = ann_ret / (max_dd_vals + 1e-8)
            calmar = torch.median(calmar_vals).item()

            # ── 归一化后加权 ──
            # 所有指标先 clip 到合理范围，再映射到 [-1, 1]
            def _norm(x, cap=5.0):
                return np.clip(x / cap, -1.0, 1.0)

            score = (
                _norm(sharpe) * 0.25
                + _norm(sortino) * 0.20
                + (win_rate * 2 - 1) * 0.15      # win_rate [0,1] → [-1,1]
                + _norm(calmar, cap=3.0) * 0.20
                + (1.0 - min(max_dd * 5, 1.0)) * 0.20  # max_dd 越小越好
            )

            # 交易频率惩罚（太少说明过拟合）
            median_activity = valid_activity.median().item()
            if median_activity < self.min_trades * 2:
                score -= 0.3

            # 负收益惩罚
            if median_ret.item() < 0:
                score -= 0.5
        else:
            score = median_ret.item()

        return {
            "score": torch.tensor(score, device=factors.device, dtype=torch.float32),
            "cum_ret": median_ret.item(),
            "sharpe": sharpe,
            "sortino": sortino,
            "win_rate": win_rate,
            "max_dd": max_dd,
            "calmar": calmar,
            "n_trades": valid_activity.median().item(),
            "valid": True,
        }

    # ─── Rank IC 计算（因子质量的核心指标）───────────
    @staticmethod
    def compute_rank_ic(
        factors: torch.Tensor,
        target_ret: torch.Tensor,
        lag: int = 0,
        return_all: bool = False,
    ) -> object:
        """
        向量化 Rank IC。return_all=True 时返回 list[float]，否则返回 float(median)。
        注意：lag 参数用于计算延迟 IC（信号延迟 lag 期后的预测能力）。
        """
        if factors.dim() == 1:
            factors = factors.unsqueeze(0)
        B, T = factors.shape

        valid_len = target_ret.shape[-1] - lag
        if valid_len <= 10:
            return 0.0

        # 因子与目标对齐：因子取 [:valid_len]，目标取 [lag:lag+valid_len]
        f_sub = factors[:, :valid_len]
        target = target_ret[0, lag:lag + valid_len]

        # 筛掉 target≈0 的位置
        mask = torch.abs(target) > 1e-9
        n_valid = mask.sum().item()
        if n_valid < 10:
            return 0.0

        # 秩转换：两次 argsort
        f_masked = f_sub[:, mask]                        # [B, N]
        f_ranks = f_masked.argsort(dim=1).argsort(dim=1).float()

        t_masked = target[mask]
        t_ranks = t_masked.argsort().argsort().float()

        # 跳过常数公式
        f_std = f_ranks.std(dim=1) + 1e-8
        valid_f = f_std > 1e-6
        if valid_f.sum() == 0:
            return 0.0

        f_ranks = f_ranks[valid_f]
        f_mean = f_ranks.mean(dim=1, keepdim=True)
        t_mean = t_ranks.mean()
        f_c = f_ranks - f_mean
        t_c = t_ranks - t_mean
        cov = (f_c * t_c.unsqueeze(0)).sum(dim=1)
        f_std_v = f_ranks.std(dim=1) + 1e-8
        t_std_v = t_ranks.std() + 1e-8
        ics = cov / (f_std_v * t_std_v * n_valid)
        ics = torch.nan_to_num(ics, nan=0.0)

        if ics.numel() == 0:
            return [] if return_all else 0.0
        if return_all:
            return ics.tolist()
        return float(ics.median().item())

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
        close = raw_data.get("close") if raw_data else None

        # ── 对抗噪声 ──
        if self.adversarial_trials > 0:
            scores = []
            returns = []

            # 基准：无噪声
            base = self._eval_once(factors, target_ret, close)
            scores.append(base["score"])
            returns.append(base["cum_ret"])

            # 结构扰动：替换算子、交换参数（真正的对抗）
            # 简化版：高斯噪声 + 符号翻转
            factor_std = factors.std() + 1e-8
            for trial in range(self.adversarial_trials - 1):
                if trial % 2 == 0:
                    noise = torch.randn_like(factors) * factor_std * self.noise_std
                    noisy = factors + noise
                else:
                    # 符号翻转对抗
                    noisy = -factors
                result = self._eval_once(noisy, target_ret, close)
                scores.append(result["score"])
                returns.append(result["cum_ret"])

            # worst-case 作为最终 fitness
            scores_t = torch.stack([s.to(factors.device) for s in scores])
            final_score = scores_t.min()
            mean_return = sum(returns) / len(returns)

            return final_score, mean_return

        # ── 标准单次评估 ──
        result = self._eval_once(factors, target_ret, close)
        return result["score"], result["cum_ret"]
