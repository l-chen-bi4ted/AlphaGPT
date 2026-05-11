"""
RiskEngine — richenlin 四级防御状态机（CEX 现货精简版）。

搬运自 richenlin/AlphaGPT strategy_manager/risk_engine.py。
移除：ModeManager、TelegramAlert、蜜罐检测、流动性查询、异步方法。
保留：RiskLevel 递进、AccountState、持仓风险检查、日回撤熔断。

用法:
    engine = RiskEngine()
    engine.add_position("BTC-USDT", entry_price=82000, amount=1.0)
    
    # 每轮检查
    actions = engine.check_position_risk("BTC-USDT", current_price=83000)
    if "stop_loss" in actions or "trailing_stop" in actions:
        close_position()
    
    # 账户级熔断
    allowed, reason = engine.check_account_risk()
    if not allowed:
        cooldown()
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from loguru import logger


class RiskLevel(Enum):
    """风险等级 — 四级渐进式防御"""
    NORMAL = "normal"          # 满额交易
    REDUCED = "reduced"        # 仓位减半
    PROTECTION = "protection"  # 只平仓不开仓
    EMERGENCY = "emergency"    # 强制全平 + 锁定


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    entry_price: float
    amount: float
    entry_time: datetime = field(default_factory=datetime.now)
    current_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    highest_price: float = 0.0
    side: str = "long"


@dataclass
class AccountState:
    """账户状态追踪"""
    total_value: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    last_trade_pnl: float = 0.0
    daily_trades: int = 0
    last_reset_date: datetime = field(default_factory=datetime.now)


class RiskEngine:
    """
    四级防御风控引擎（richenlin 移植版）。

    NORMAL → REDUCED(日回撤>10%) → PROTECTION(连亏4次) → EMERGENCY(日回撤>20%)
    每个等级自动调整交易规模乘数。
    """

    def __init__(
        self,
        stop_loss_pct: float = -0.05,
        trailing_stop_pct: float = 0.06,
        take_profit_pcts: Optional[List[float]] = None,
        max_holding_hours: int = 72,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pcts = take_profit_pcts or [0.10, 0.20]
        self.max_holding_hours = max_holding_hours

        self.risk_level = RiskLevel.NORMAL
        self.account_state = AccountState()
        self.positions: Dict[str, Position] = {}

    # ─── 持仓管理 ──────────────────────────────────

    def add_position(self, symbol: str, entry_price: float, amount: float):
        """登记新持仓。"""
        pos = Position(
            symbol=symbol, entry_price=entry_price, amount=amount,
            highest_price=entry_price,
        )
        self.positions[symbol] = pos
        logger.info(f"[Risk] Position added: {symbol} {amount} @ {entry_price}")

    def remove_position(self, symbol: str):
        """移除持仓（平仓后调用）。"""
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            logger.info(f"[Risk] Position closed: {symbol} PnL={pos.pnl_pct:.2%}")

    # ─── 持仓风险检查 (Layer 2) ───────────────────

    def check_position_risk(self, symbol: str, current_price: float) -> List[str]:
        """
        检查单个持仓风险。

        Returns:
            动作列表：['stop_loss', 'trailing_stop', 'take_profit_1', 'time_out']
        """
        if symbol not in self.positions:
            return []

        pos = self.positions[symbol]
        pos.current_price = current_price

        # 更新 PnL
        pos.pnl = (current_price - pos.entry_price) * pos.amount
        pos.pnl_pct = (current_price - pos.entry_price) / pos.entry_price
        pos.highest_price = max(pos.highest_price, current_price)

        actions = []

        # 硬止损
        if pos.pnl_pct <= self.stop_loss_pct:
            actions.append("stop_loss")
            logger.warning(f"[Risk] {symbol} STOP LOSS: {pos.pnl_pct:.2%}")

        # 追踪止损（曾在入场价上方 2% 以上 → 激活）
        if pos.highest_price > pos.entry_price * 1.02:
            trail_price = pos.highest_price * (1 - self.trailing_stop_pct)
            if current_price <= trail_price:
                actions.append("trailing_stop")
                logger.info(f"[Risk] {symbol} TRAILING: peak={pos.highest_price:.1f} now={current_price:.1f} trail={trail_price:.1f}")

        # 分批止盈
        for i, tp_pct in enumerate(self.take_profit_pcts):
            if pos.pnl_pct >= tp_pct:
                actions.append(f"take_profit_{i+1}")
                logger.info(f"[Risk] {symbol} TAKE PROFIT {i+1}: {pos.pnl_pct:.2%}")

        # 时间止损
        holding = datetime.now() - pos.entry_time
        if holding > timedelta(hours=self.max_holding_hours):
            actions.append("time_out")
            logger.warning(f"[Risk] {symbol} TIME OUT: {holding}")

        return actions

    # ─── 账户级保护 (Layer 3) ──────────────────────

    def check_account_risk(self) -> Tuple[bool, str]:
        """
        检查账户级风险，自动升降风险等级。

        Returns:
            (allowed, reason): 是否允许开新仓
        """
        daily = self.account_state.daily_pnl_pct
        old_level = self.risk_level

        # 逐级递进
        if daily <= -0.20:
            self.risk_level = RiskLevel.EMERGENCY
        elif daily <= -0.10:
            self.risk_level = RiskLevel.PROTECTION
        elif daily <= -0.05:
            self.risk_level = RiskLevel.REDUCED

        # 连亏升级
        if self.account_state.consecutive_losses >= 4:
            self.risk_level = RiskLevel.PROTECTION
            logger.warning(f"[Risk] {self.account_state.consecutive_losses} consecutive losses → PROTECTION")

        # 恢复逻辑
        if self.risk_level == RiskLevel.REDUCED and daily > -0.03 and self.account_state.consecutive_wins >= 2:
            self.risk_level = RiskLevel.NORMAL
            logger.info("[Risk] Downgraded to NORMAL")
        elif self.risk_level == RiskLevel.PROTECTION and daily > -0.05 and self.account_state.consecutive_wins >= 3:
            self.risk_level = RiskLevel.REDUCED
            logger.info("[Risk] Downgraded to REDUCED")

        if old_level != self.risk_level:
            logger.warning(f"[Risk] Level: {old_level.value} → {self.risk_level.value}")

        # 根据等级决定是否允许开仓
        if self.risk_level == RiskLevel.EMERGENCY:
            return False, "EMERGENCY: all trading halted"
        if self.risk_level == RiskLevel.PROTECTION:
            return False, "PROTECTION: close-only mode"
        if self.risk_level == RiskLevel.REDUCED:
            return True, "REDUCED: half position only"

        return True, "NORMAL"

    def get_risk_multiplier(self) -> float:
        """根据当前风险等级返回仓位乘数。"""
        return {
            RiskLevel.NORMAL: 1.0,
            RiskLevel.REDUCED: 0.5,
            RiskLevel.PROTECTION: 0.0,
            RiskLevel.EMERGENCY: 0.0,
        }[self.risk_level]

    # ─── 状态更新 ──────────────────────────────────

    def update_account_state(self, pnl: float, trade_count: int = 1):
        """每笔交易后更新账户状态。"""
        self.account_state.daily_trades += trade_count
        self.account_state.daily_pnl += pnl
        self.account_state.last_trade_pnl = pnl

        if pnl > 0:
            self.account_state.consecutive_wins += 1
            self.account_state.consecutive_losses = 0
        else:
            self.account_state.consecutive_losses += 1
            self.account_state.consecutive_wins = 0

        logger.info(
            f"[Risk] Trade recorded: PnL={pnl:+.2f} "
            f"streak={'W' if pnl>0 else 'L'}{self.account_state.consecutive_wins if pnl>0 else self.account_state.consecutive_losses}"
        )

    def update_daily_pnl(self, current_equity: float):
        """更新日内权益和日回撤百分比。"""
        if self.account_state.total_value > 0:
            self.account_state.daily_pnl = current_equity - self.account_state.total_value
            self.account_state.daily_pnl_pct = self.account_state.daily_pnl / self.account_state.total_value

    def reset_daily_state(self, new_equity: float):
        """每日重置。"""
        self.risk_level = RiskLevel.NORMAL
        self.account_state = AccountState(
            total_value=new_equity,
            last_reset_date=datetime.now(),
        )
        logger.info(f"[Risk] Daily reset. Equity: {new_equity:.0f} level: {self.risk_level.value}")

    # ─── 总览 ──────────────────────────────────────

    def get_summary(self) -> Dict:
        """返回风控状态摘要。"""
        return {
            "risk_level": self.risk_level.value,
            "multiplier": self.get_risk_multiplier(),
            "daily_pnl_pct": f"{self.account_state.daily_pnl_pct:.2%}",
            "daily_trades": self.account_state.daily_trades,
            "streak": f"W{self.account_state.consecutive_wins}" if self.account_state.consecutive_wins > 0
                      else f"L{self.account_state.consecutive_losses}",
            "positions": len(self.positions),
        }
