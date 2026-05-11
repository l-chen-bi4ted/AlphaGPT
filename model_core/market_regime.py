"""
MarketRegime v3 — 市场状态检测（无未来函数版）。

v3 修复：
- 消除 np.roll 导致的未来函数（第一个元素用了序列末尾）
- ATR 历史使用 collections.deque，避免 list.pop(0) 的 O(n) 开销
- 数据不足时默认返回 RANGING（更保守）
- 不再直接修改 RiskEngine 属性，改为返回参数 dict 由调用方决定是否应用
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict
from collections import deque
import numpy as np
from loguru import logger


class Regime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class RegimeParams:
    """不同市场状态下的风控参数。"""
    stop_loss_pct: float
    trailing_stop_pct: float
    take_profit_pcts: list
    position_multiplier: float


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
    市场状态检测器 v3。

    用法:
        regime = MarketRegime()
        state = regime.detect(high, low, close, period=14)
        params = regime.get_params(state)
        # 由调用方决定是否应用：
        # risk_engine.stop_loss_pct = params.stop_loss_pct
    """

    def __init__(
        self,
        adx_threshold: float = 25,
        atr_multiplier: float = 2.5,
        atr_history_len: int = 100,
    ):
        self.adx_threshold = adx_threshold
        self.atr_multiplier = atr_multiplier
        self.current_regime = Regime.RANGING  # 默认保守
        self._atr_history: deque = deque(maxlen=atr_history_len)

    @staticmethod
    def _shift_prev(arr: np.ndarray) -> np.ndarray:
        """前向偏移：out[0] = arr[0], out[t] = arr[t-1]。无未来函数。"""
        out = np.empty_like(arr)
        out[0] = arr[0]
        out[1:] = arr[:-1]
        return out

    @staticmethod
    def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """True Range: max(H-L, |H-C_prev|, |L-C_prev|)"""
        prev_close = MarketRegime._shift_prev(close)
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(np.maximum(tr1, tr2), tr3)

    @staticmethod
    def _smooth(values: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing。"""
        result = np.zeros_like(values)
        if len(values) < period:
            return result
        result[period - 1] = values[:period].mean()
        for i in range(period, len(values)):
            result[i] = (result[i - 1] * (period - 1) + values[i]) / period
        return result

    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """计算最新 ADX 值。"""
        if len(close) < period * 2:
            return 15.0  # 数据不足，返回保守值（低于 20）

        tr = MarketRegime._true_range(high, low, close)
        atr = MarketRegime._smooth(tr, period)

        prev_high = MarketRegime._shift_prev(high)
        prev_low = MarketRegime._shift_prev(low)

        up_move = high - prev_high
        down_move = prev_low - low
        up_move[0] = down_move[0] = 0

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_di = 100 * MarketRegime._smooth(plus_dm, period) / (atr + 1e-8)
        minus_di = 100 * MarketRegime._smooth(minus_dm, period) / (atr + 1e-8)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx_vals = MarketRegime._smooth(dx, period)

        return float(adx_vals[-1])

    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """计算 ATR。"""
        if len(close) < 2:
            return 0.0
        tr = MarketRegime._true_range(high, low, close)
        return float(np.mean(tr[-min(period, len(tr)):]))

    @staticmethod
    def atr_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """ATR 占当前价格的百分比。"""
        a = MarketRegime.atr(high, low, close, period)
        last_close = close[-1]
        if last_close <= 0 or not np.isfinite(last_close):
            return 0.0
        return a / last_close

    def detect(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> Regime:
        """检测当前市场状态。"""
        if len(close) < period * 2:
            logger.debug("MarketRegime: insufficient data, defaulting to RANGING")
            self.current_regime = Regime.RANGING
            return self.current_regime

        adx_val = self.adx(high, low, close, period)
        atr_pct_val = self.atr_pct(high, low, close, period)

        # 更新 ATR 历史
        if np.isfinite(atr_pct_val):
            self._atr_history.append(atr_pct_val)

        # 波动率突变检测
        atr_spike = False
        if len(self._atr_history) >= 20:
            hist = np.array(list(self._atr_history))
            avg_atr = np.mean(hist[-20:])
            std_atr = np.std(hist[-20:])
            # 用均值 + 2*std 作为阈值，比简单倍数更稳健
            if avg_atr > 0 and atr_pct_val > avg_atr + self.atr_multiplier * std_atr:
                atr_spike = True

        # 状态判定
        if atr_spike:
            self.current_regime = Regime.VOLATILE
        elif adx_val < 20:
            self.current_regime = Regime.RANGING
        elif adx_val > self.adx_threshold:
            self.current_regime = Regime.TRENDING
        else:
            # 20 <= adx <= 25，保持上一状态（避免频繁切换）
            pass

        logger.debug(
            f"MarketRegime: ADX={adx_val:.1f} ATR%={atr_pct_val:.4f} → {self.current_regime.value}"
        )
        return self.current_regime

    def get_params(self, regime: Optional[Regime] = None) -> RegimeParams:
        """获取当前市场状态对应的风控参数。"""
        r = regime or self.current_regime
        return DEFAULT_REGIME_PARAMS.get(r, DEFAULT_REGIME_PARAMS[Regime.RANGING])

    def get_params_dict(self, regime: Optional[Regime] = None) -> Dict:
        """返回可序列化的参数字典，供调用方应用。"""
        p = self.get_params(regime)
        return {
            "stop_loss_pct": p.stop_loss_pct,
            "trailing_stop_pct": p.trailing_stop_pct,
            "take_profit_pcts": p.take_profit_pcts,
            "position_multiplier": p.position_multiplier,
        }
