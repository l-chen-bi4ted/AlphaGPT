"""
MarketRegime — 市场状态检测（richenlin 移植版）。

基于 ADX（趋势强度）和 ATR（波动率）判断当前市场：
  TRENDING: ADX > 25, 明确方向
  RANGING:  ADX < 20, 横盘震荡
  VOLATILE: ATR 异常放大, 剧烈波动

联动 RiskEngine:
  TRENDING → 宽止损 + 大止盈（拿趋势）
  RANGING  → 紧止损 + 快止盈（防洗盘）
  VOLATILE → 减仓 50% + 宽止损（防插针）
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class Regime(Enum):
    TRENDING = "trending"   # 趋势市：ADX > 25
    RANGING = "ranging"      # 震荡市：ADX < 20
    VOLATILE = "volatile"    # 高波动：ATR 飙升


@dataclass
class RegimeParams:
    """不同市场状态下的风控参数。"""
    stop_loss_pct: float
    trailing_stop_pct: float
    take_profit_pcts: list
    position_multiplier: float  # 仓位乘数


# ─── 默认参数映射 ────────────────────────────

DEFAULT_REGIME_PARAMS = {
    Regime.TRENDING: RegimeParams(
        stop_loss_pct=-0.05,
        trailing_stop_pct=0.06,
        take_profit_pcts=[0.10, 0.20],
        position_multiplier=1.0,
    ),
    Regime.RANGING: RegimeParams(
        stop_loss_pct=-0.03,
        trailing_stop_pct=0.04,
        take_profit_pcts=[0.05, 0.10],
        position_multiplier=0.7,
    ),
    Regime.VOLATILE: RegimeParams(
        stop_loss_pct=-0.08,
        trailing_stop_pct=0.10,
        take_profit_pcts=[0.15, 0.30],
        position_multiplier=0.5,
    ),
}


class MarketRegime:
    """
    市场状态检测器。

    用法:
        regime = MarketRegime()
        state = regime.detect(high, low, close, period=14)
        params = regime.get_params(state)
    """

    def __init__(self, adx_threshold: float = 25, atr_multiplier: float = 2.5):
        self.adx_threshold = adx_threshold
        self.atr_multiplier = atr_multiplier
        self.current_regime = Regime.TRENDING  # 默认乐观
        self._atr_history = []  # 用于波动率突变检测

    # ─── ADX 计算 ─────────────────────────────────

    @staticmethod
    def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """True Range: max(H-L, |H-C_prev|, |L-C_prev|)"""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(np.maximum(tr1, tr2), tr3)

    @staticmethod
    def _smooth(values: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing (指数移动平均变体)。"""
        result = np.zeros_like(values)
        result[period - 1] = values[:period].mean()
        for i in range(period, len(values)):
            result[i] = (result[i - 1] * (period - 1) + values[i]) / period
        return result

    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """
        计算 ADX（平均趋向指数）。

        Returns:
            最新的 ADX 值（0-100）
        """
        if len(close) < period * 2:
            return 20.0  # 数据不足，默认中性

        tr = MarketRegime._true_range(high, low, close)
        atr = MarketRegime._smooth(tr, period)

        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = down_move[0] = 0

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_di = 100 * MarketRegime._smooth(plus_dm, period) / (atr + 1e-8)
        minus_di = 100 * MarketRegime._smooth(minus_dm, period) / (atr + 1e-8)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx_vals = MarketRegime._smooth(dx, period)

        return float(adx_vals[-1])

    # ─── ATR 计算 ─────────────────────────────────

    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """计算 ATR（平均真实波幅）。"""
        if len(close) < period:
            return 0.0
        tr = MarketRegime._true_range(high, low, close)
        return float(np.mean(tr[-period:]))

    @staticmethod
    def atr_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """ATR 占当前价格的百分比。"""
        a = MarketRegime.atr(high, low, close, period)
        return a / close[-1] if close[-1] > 0 else 0.0

    # ─── 状态检测 ─────────────────────────────────

    def detect(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> Regime:
        """
        检测当前市场状态。

        Args:
            high/low/close: numpy 数组（最近 N 根 K 线）
            period: ADX/ATR 计算周期

        Returns:
            Regime 枚举
        """
        if len(close) < period * 2:
            return Regime.TRENDING  # 数据不足，默认趋势

        adx_val = self.adx(high, low, close, period)
        atr_pct_val = self.atr_pct(high, low, close, period)

        # 追踪 ATR 历史，检测波动率突变
        self._atr_history.append(atr_pct_val)
        if len(self._atr_history) > 100:
            self._atr_history.pop(0)

        # 波动率突变检测
        atr_spike = False
        if len(self._atr_history) >= 20:
            avg_atr = np.mean(self._atr_history[-20:])
            if avg_atr > 0 and atr_pct_val > avg_atr * self.atr_multiplier:
                atr_spike = True

        # 状态判定
        if atr_spike:
            self.current_regime = Regime.VOLATILE
        elif adx_val < 20:
            self.current_regime = Regime.RANGING
        else:
            self.current_regime = Regime.TRENDING

        return self.current_regime

    # ─── 参数获取 ─────────────────────────────────

    def get_params(self, regime: Optional[Regime] = None) -> RegimeParams:
        """获取当前市场状态对应的风控参数。"""
        r = regime or self.current_regime
        return DEFAULT_REGIME_PARAMS.get(r, DEFAULT_REGIME_PARAMS[Regime.TRENDING])

    def apply_to_risk_engine(self, risk_engine, regime: Optional[Regime] = None):
        """
        将市场状态参数应用到 RiskEngine。

        直接修改 RiskEngine 的止损/止盈/追踪参数。
        """
        params = self.get_params(regime)
        risk_engine.stop_loss_pct = params.stop_loss_pct
        risk_engine.trailing_stop_pct = params.trailing_stop_pct
        risk_engine.take_profit_pcts = params.take_profit_pcts
        return params.position_multiplier
