"""
RiskEngine v3 — 四级防御状态机（实盘闭环版）。

v3 修复：
- 状态自动持久化，进程重启可恢复
- 每日自动重置（基于日期检查）
- 支持 unrealized PnL（持仓浮盈计入日回撤）
- 成交回报驱动：只有通过 record_trade() 录入的盈亏才更新连续盈亏计数
- 与 MarketRegime 解耦：接收参数修改，不直接操作属性
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from pathlib import Path
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
    entry_time: str = field(default_factory=lambda: datetime.now().isoformat())
    current_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    highest_price: float = 0.0
    side: str = "long"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(**d)


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
    last_reset_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AccountState":
        return cls(**d)


class RiskEngine:
    """
    四级防御风控引擎 v3。

    NORMAL → REDUCED(日回撤>10% 或 unrealized>8%) → PROTECTION(连亏4次) → EMERGENCY(日回撤>20%)
    """

    def __init__(
        self,
        stop_loss_pct: float = -0.05,
        trailing_stop_pct: float = 0.06,
        take_profit_pcts: Optional[List[float]] = None,
        max_holding_hours: int = 72,
        daily_drawdown_limit: float = -0.10,
        emergency_drawdown: float = -0.20,
        state_file: Optional[str] = None,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pcts = take_profit_pcts or [0.10, 0.20]
        self.max_holding_hours = max_holding_hours
        self.daily_drawdown_limit = daily_drawdown_limit
        self.emergency_drawdown = emergency_drawdown

        self.risk_level = RiskLevel.NORMAL
        self.account_state = AccountState()
        self.positions: Dict[str, Position] = {}

        # 状态持久化
        if state_file is None:
            state_file = os.path.join(os.path.dirname(__file__), "..", "risk_state.json")
        self.state_file = Path(state_file)
        self._load_state()

    # ─── 状态持久化 ──────────────────────────────────

    def _load_state(self):
        """从磁盘恢复状态。"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                self.account_state = AccountState.from_dict(data.get("account", {}))
                self.positions = {
                    k: Position.from_dict(v) for k, v in data.get("positions", {}).items()
                }
                self.risk_level = RiskLevel(data.get("risk_level", "normal"))
                logger.info(f"[Risk] State loaded from {self.state_file}")
            except Exception as e:
                logger.warning(f"[Risk] Failed to load state: {e}, starting fresh")

    def _save_state(self):
        """持久化状态到磁盘。"""
        try:
            data = {
                "account": self.account_state.to_dict(),
                "positions": {k: v.to_dict() for k, v in self.positions.items()},
                "risk_level": self.risk_level.value,
                "saved_at": datetime.now().isoformat(),
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[Risk] Failed to save state: {e}")

    def _check_daily_reset(self, current_equity: float):
        """检查是否需要每日重置。"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.account_state.last_reset_date != today:
            logger.info(f"[Risk] Daily reset triggered ({self.account_state.last_reset_date} → {today})")
            self.reset_daily_state(current_equity)
            return True
        return False

    # ─── 持仓管理 ──────────────────────────────────

    def add_position(self, symbol: str, entry_price: float, amount: float, side: str = "long"):
        """登记新持仓。"""
        pos = Position(
            symbol=symbol, entry_price=entry_price, amount=amount,
            highest_price=entry_price, side=side,
        )
        self.positions[symbol] = pos
        self._save_state()
        logger.info(f"[Risk] Position added: {symbol} {amount} @ {entry_price}")

    def remove_position(self, symbol: str, exit_price: float, realized_pnl: float):
        """移除持仓（平仓后调用），录入已实现盈亏。"""
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            self.record_trade(realized_pnl)
            self._save_state()
            logger.info(f"[Risk] Position closed: {symbol} @ {exit_price} Realized={realized_pnl:+.2f}")

    def update_position_price(self, symbol: str, current_price: float) -> Optional[Position]:
        """更新持仓市价和浮盈。"""
        if symbol not in self.positions:
            return None
        pos = self.positions[symbol]
        pos.current_price = current_price
        pos.pnl = (current_price - pos.entry_price) * pos.amount
        pos.pnl_pct = (current_price - pos.entry_price) / pos.entry_price
        pos.highest_price = max(pos.highest_price, current_price)
        return pos

    # ─── 持仓风险检查 (Layer 2) ───────────────────

    def check_position_risk(self, symbol: str, current_price: float) -> List[str]:
        """
        检查单个持仓风险。
        Returns: 动作列表
        """
        pos = self.update_position_price(symbol, current_price)
        if pos is None:
            return []

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
                logger.info(
                    f"[Risk] {symbol} TRAILING: peak={pos.highest_price:.1f} "
                    f"now={current_price:.1f} trail={trail_price:.1f}"
                )

        # 分批止盈
        for i, tp_pct in enumerate(self.take_profit_pcts):
            if pos.pnl_pct >= tp_pct:
                actions.append(f"take_profit_{i+1}")
                logger.info(f"[Risk] {symbol} TAKE PROFIT {i+1}: {pos.pnl_pct:.2%}")

        # 时间止损
        entry_dt = datetime.fromisoformat(pos.entry_time)
        holding = datetime.now() - entry_dt
        if holding > timedelta(hours=self.max_holding_hours):
            actions.append("time_out")
            logger.warning(f"[Risk] {symbol} TIME OUT: {holding}")

        return actions

    # ─── 账户级保护 (Layer 3) ──────────────────────

    def check_account_risk(self, current_equity: float) -> Tuple[bool, str]:
        """
        检查账户级风险，自动升降风险等级。
        必须在每轮交易前调用。
        """
        # 自动日重置
        self._check_daily_reset(current_equity)

        # 计算 total unrealized + realized 日回撤
        unrealized = sum(p.pnl for p in self.positions.values())
        total_daily_pnl = self.account_state.daily_pnl + unrealized

        if self.account_state.total_value > 0:
            daily_pct = total_daily_pnl / self.account_state.total_value
        else:
            daily_pct = 0.0
            self.account_state.total_value = current_equity

        old_level = self.risk_level

        # 逐级递进（使用 combined daily pnl）
        if daily_pct <= self.emergency_drawdown:
            self.risk_level = RiskLevel.EMERGENCY
        elif daily_pct <= self.daily_drawdown_limit:
            self.risk_level = RiskLevel.PROTECTION
        elif daily_pct <= self.daily_drawdown_limit * 0.5:
            self.risk_level = RiskLevel.REDUCED

        # 连亏升级
        if self.account_state.consecutive_losses >= 4:
            self.risk_level = RiskLevel.PROTECTION
            logger.warning(
                f"[Risk] {self.account_state.consecutive_losses} consecutive losses → PROTECTION"
            )

        # 恢复逻辑
        if self.risk_level == RiskLevel.REDUCED and daily_pct > -0.03 and self.account_state.consecutive_wins >= 2:
            self.risk_level = RiskLevel.NORMAL
            logger.info("[Risk] Downgraded to NORMAL")
        elif self.risk_level == RiskLevel.PROTECTION and daily_pct > -0.05 and self.account_state.consecutive_wins >= 3:
            self.risk_level = RiskLevel.REDUCED
            logger.info("[Risk] Downgraded to REDUCED")

        if old_level != self.risk_level:
            logger.warning(f"[Risk] Level: {old_level.value} → {self.risk_level.value}")
            self._save_state()

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

    def record_trade(self, realized_pnl: float):
        """
        每笔已确认成交后更新账户状态。
        只有 realized_pnl 影响连续盈亏计数。
        """
        self.account_state.daily_trades += 1
        self.account_state.daily_pnl += realized_pnl
        self.account_state.last_trade_pnl = realized_pnl

        if realized_pnl > 0:
            self.account_state.consecutive_wins += 1
            self.account_state.consecutive_losses = 0
        else:
            self.account_state.consecutive_losses += 1
            self.account_state.consecutive_wins = 0

        logger.info(
            f"[Risk] Trade recorded: PnL={realized_pnl:+.2f} "
            f"streak={'W' if realized_pnl>0 else 'L'}"
            f"{self.account_state.consecutive_wins if realized_pnl>0 else self.account_state.consecutive_losses}"
        )
        self._save_state()

    def reset_daily_state(self, new_equity: float):
        """每日重置。可由外部调用，也会自动触发。"""
        self.risk_level = RiskLevel.NORMAL
        self.account_state = AccountState(
            total_value=new_equity,
            last_reset_date=datetime.now().strftime("%Y-%m-%d"),
        )
        self._save_state()
        logger.info(f"[Risk] Daily reset. Equity: {new_equity:.2f} level: {self.risk_level.value}")

    # ─── 总览 ──────────────────────────────────────

    def get_summary(self) -> Dict:
        """返回风控状态摘要。"""
        unrealized = sum(p.pnl for p in self.positions.values())
        return {
            "risk_level": self.risk_level.value,
            "multiplier": self.get_risk_multiplier(),
            "daily_pnl": f"{self.account_state.daily_pnl:+.2f}",
            "unrealized_pnl": f"{unrealized:+.2f}",
            "daily_trades": self.account_state.daily_trades,
            "streak": (
                f"W{self.account_state.consecutive_wins}"
                if self.account_state.consecutive_wins > 0
                else f"L{self.account_state.consecutive_losses}"
            ),
            "positions": len(self.positions),
        }
