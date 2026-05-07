"""
CEX 回测引擎 — 替代 MemeBacktest。

适用于 OKX 等中心化交易所：无滑点建模、无流动性门槛、费率低。
"""

import torch
from .config import ModelConfig


class CEXBacktest:
    """
    CEX 现货回测。

    Args:
        fee_rate: 单边手续费率（默认 0.001 = 0.1%）
        trade_size: 每笔名义交易金额（USD）
        signal_threshold: sigmoid 信号 > 此值才入场
        min_trades: 最少交易次数，低于此则判定为无效公式
    """

    def __init__(
        self,
        fee_rate: float = None,
        trade_size: float = None,
        signal_threshold: float = None,
        min_trades: int = 5,
    ):
        self.fee_rate = fee_rate if fee_rate is not None else ModelConfig.BASE_FEE
        self.trade_size = trade_size if trade_size is not None else ModelConfig.TRADE_SIZE_USD
        self.threshold = signal_threshold if signal_threshold is not None else ModelConfig.SIGNAL_THRESHOLD
        self.min_trades = min_trades

    def evaluate(
        self,
        factors: torch.Tensor,
        raw_data: dict,
        target_ret: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        评估一组因子在历史数据上的表现。

        Args:
            factors: [T] 或 [B, T] — 因子时间序列
            raw_data: 原始数据 dict（未使用，保留接口兼容）
            target_ret: [1, T] — 下期收益率

        Returns:
            score: 中位数 fitness（用于 RL reward）
            mean_return: 平均累计收益
        """
        # 统一为 [B, T]
        if factors.dim() == 1:
            factors = factors.unsqueeze(0)
        B, T = factors.shape

        # 信号 → 持仓权重
        signal = torch.sigmoid(factors)  # [B, T]
        position = (signal > self.threshold).float()  # [B, T]

        # 换手 → 手续费
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * self.fee_rate

        # PnL
        # target_ret shape: [1, T] → broadcast to [B, T]
        gross_pnl = position * target_ret  # [B, T]
        net_pnl = gross_pnl - tx_cost

        # 累计收益
        cum_ret = net_pnl.sum(dim=1)  # [B]

        # 大幅回撤惩罚
        big_dd = (net_pnl < -0.02).float().sum(dim=1)

        # 交易活跃度检查
        activity = position.sum(dim=1)

        # 综合打分
        score = cum_ret - (big_dd * 1.0)
        score = torch.where(
            activity < self.min_trades,
            torch.tensor(-10.0, device=score.device),
            score,
        )

        final_fitness = torch.median(score)
        return final_fitness, cum_ret.mean().item()
