"""
AlphaGPT + OKX 模拟盘运行器 v2 — 风控装甲版。

流程：
  1. 加载训练好的因子公式
  2. 每小时拉取最新 K 线，计算 z-score 信号
  3. 风控先行：硬止损 -5% / 追踪止损 6% / 单日熔断 15%
  4. 信号交易：signal > 0.5 买入，signal < 0.3 卖出
  5. 熔断冷却：风控触发后暂停 24 小时
"""

import asyncio
import json
import os
import time
from collections import deque
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import torch
import numpy as np
from loguru import logger

from model_core.vm import StackVM
from model_core.config import ModelConfig
from okx_data import fetch_all_candles
from okx_executor import OKXExecutor


# ─── 风控模式（改这一行切换） ──────────────────
# "conservative": 紧止损 + 低回撤 + 冷却 8h（默认推荐）
# "aggressive":   宽止损 + 高回撤 + 短冷却 4h（牛市）
RISK_MODE = "conservative"

# ─── 风控参数（richenlin 双模式设计）───────────
RISK_PROFILES = {
    "conservative": {
        "hard_stop":      -0.05,    # 硬止损 -5%
        "trailing_stop":   0.06,    # 追踪止损 6%
        "trailing_activation": 0.03,  # 先盈利 3% 才启用追踪
        "daily_max_dd":    0.10,    # 单日熔断 10%
        "cooldown_hours":  8,       # 冷却 8h
    },
    "aggressive": {
        "hard_stop":      -0.08,    # 硬止损 -8%
        "trailing_stop":   0.10,    # 追踪止损 10%
        "trailing_activation": 0.05,  # 先盈利 5% 才启用追踪
        "daily_max_dd":    0.20,    # 单日熔断 20%
        "cooldown_hours":  4,       # 冷却 4h
    },
}

def _cfg(key: str):
    """读取当前风险模式的配置值。"""
    return RISK_PROFILES[RISK_MODE][key]


class LiveRunner:
    def __init__(
        self,
        formula_path: Optional[str] = None,
        inst_id: Optional[str] = None,
        bar: str = "1H",
        demo: bool = True,
    ):
        self.inst_id = inst_id or ModelConfig.INST_ID
        self.bar = bar
        self.demo = demo
        self.base_ccy = self.inst_id.split("-")[0]

        # 加载因子公式
        if formula_path is None:
            prefix = f"{self.inst_id.replace('-','')}_{self.bar}"
            formula_path = os.path.join(ModelConfig.SAVE_DIR, f"{prefix}_formula.json")

        with open(formula_path, "r") as f:
            data = json.load(f)
            self.formula = data if isinstance(data, list) else data.get("formula")
        logger.success(f"Loaded formula: {self.formula}")

        self.vm = StackVM()
        self.executor = OKXExecutor(demo=demo)

        # ── 风控状态 ──
        self.position = None         # "long" | None
        self.entry_price = 0.0       # 入场价（用于 PnL 计算）
        self.peak_price = 0.0        # 入场后最高价（追踪止损基准）
        self.last_risk_event = 0.0   # 上次风控触发时间戳
        self.daily_start_equity = 0.0  # 当日起始权益（熔断基准）
        self.daily_start_time = 0.0  # 当日起始时间

        # 启动对账
        self._reconcile()

    # ─── 启动对账 ──────────────────────────────────
    def _reconcile(self):
        """检查实际持仓，设置初始状态。"""
        try:
            bal = self.executor.get_balance()
            btc = bal.get(self.base_ccy, 0)
            if btc > 0.0001:
                self.position = "long"
                ticker = OKXExecutor.get_ticker(self.inst_id)
                current_price = ticker.get("last", 0)
                self.entry_price = current_price  # 无法得知真实成本，用现价近似
                self.peak_price = current_price
                self.daily_start_equity = self._total_equity()
                self.daily_start_time = time.time()
                logger.info(
                    f"Reconciled: {btc:.6f} {self.base_ccy} @ ~{current_price:.1f} "
                    f"(unknown cost, using current)"
                )
        except Exception as e:
            logger.warning(f"Reconcile failed: {e}")

    # ─── 信号计算 ──────────────────────────────────
    def _compute_signal(self) -> float:
        """拉取 K 线 → 执行公式 → 滚动 z-score → 0~1 信号。"""
        df = fetch_all_candles(self.inst_id, self.bar, limit=200)
        if df.empty or len(df) < 50:
            return 0.5

        device = ModelConfig.DEVICE
        close = torch.tensor(df["close"].values[-200:], dtype=torch.float32, device=device).unsqueeze(0)
        open_ = torch.tensor(df["open"].values[-200:], dtype=torch.float32, device=device).unsqueeze(0)
        high  = torch.tensor(df["high"].values[-200:], dtype=torch.float32, device=device).unsqueeze(0)
        low   = torch.tensor(df["low"].values[-200:], dtype=torch.float32, device=device).unsqueeze(0)
        vol   = torch.tensor(df["vol"].values[-200:], dtype=torch.float32, device=device).unsqueeze(0)

        raw = {
            "close": close, "open": open_, "high": high, "low": low, "volume": vol,
            "liquidity": torch.full_like(vol, 1e9), "fdv": torch.full_like(vol, 1e12),
        }
        from model_core.factors import FeatureEngineer
        feat = FeatureEngineer.compute_features(raw)
        res = self.vm.execute(self.formula, feat)
        if res is None:
            return 0.5

        raw_seq = res[0, :].cpu().numpy()
        current = float(raw_seq[-1])

        if not hasattr(self, "_signal_hist"):
            self._signal_hist = deque(maxlen=200)
        self._signal_hist.append(current)

        if len(self._signal_hist) < 30:
            return 0.5

        arr = np.array(self._signal_hist)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median)) + 1e-8
        z = 0.6745 * (current - median) / mad
        signal = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(signal, 0.01, 0.99))

    # ─── 风控检查 ──────────────────────────────────
    def _check_risk(self) -> Optional[str]:
        """返回触发原因字符串，无风险则返回 None。"""
        now = time.time()
        hard_stop_pct = _cfg("hard_stop")
        trail_pct = _cfg("trailing_stop")
        trail_act = _cfg("trailing_activation")
        daily_dd_pct = _cfg("daily_max_dd")
        cooldown_h = _cfg("cooldown_hours")

        # 1. 熔断冷却中
        if self.last_risk_event > 0:
            elapsed = (now - self.last_risk_event) / 3600
            if elapsed < cooldown_h:
                remaining = cooldown_h - elapsed
                logger.info(f"[COOLDOWN] {remaining:.1f}h remaining, skipping trade")
                return "cooldown"
            else:
                logger.info("[COOLDOWN] expired, resuming")
                self.last_risk_event = 0.0
                self.daily_start_time = now
                self.daily_start_equity = self._total_equity()

        if self.position != "long":
            return None  # 无持仓，不检查 PnL 风控

        # 2. 当前价格
        ticker = OKXExecutor.get_ticker(self.inst_id)
        current_price = ticker.get("last", 0)
        if current_price <= 0:
            return None

        # 3. PnL
        pnl_pct = (current_price - self.entry_price) / self.entry_price
        self.peak_price = max(self.peak_price, current_price)

        # 3a. 硬止损
        if pnl_pct <= hard_stop_pct:
            logger.error(
                f"[HARD STOP] PnL {pnl_pct:.2%} <= {hard_stop_pct:.2%} "
                f"(entry={self.entry_price:.1f} now={current_price:.1f})"
            )
            return "hard_stop"

        # 3b. 追踪止损（需先盈利激活）
        if pnl_pct >= trail_act:
            drawdown_from_peak = (current_price - self.peak_price) / self.peak_price
            if drawdown_from_peak <= -trail_pct:
                logger.error(
                    f"[TRAILING STOP] DD from peak {drawdown_from_peak:.2%} <= "
                    f"{-trail_pct:.2%} "
                    f"(peak={self.peak_price:.1f} now={current_price:.1f})"
                )
                return "trailing_stop"
        # 4. 单日熔断
        equity = self._total_equity()
        if self.daily_start_equity > 0:
            daily_max = daily_dd_pct
            daily_pnl = (equity - self.daily_start_equity) / self.daily_start_equity
            if daily_pnl <= -daily_max:
                logger.error(
                    f"[DAILY DD] {daily_pnl:.2%} <= {-daily_max:.2%} "
                    f"(start={self.daily_start_equity:.0f} now={equity:.0f})"
                )
                return "daily_dd"

        return None

    def _trigger_risk_response(self, reason: str):
        """风控响应：平仓 + 记录 + 熔断冷却。"""
        cooldown_h = _cfg("cooldown_hours")
        logger.critical(f"[RISK EVENT] {reason} — closing position + cooldown {cooldown_h}h")
        self._close_position()
        self.position = None
        self.last_risk_event = time.time()

    # ─── 资产计算 ──────────────────────────────────
    def _total_equity(self) -> float:
        """USDT + BTC 市值。"""
        try:
            bal = self.executor.get_balance()
            btc = bal.get(self.base_ccy, 0)
            usdt = bal.get("USDT", 0)
            ticker = OKXExecutor.get_ticker(self.inst_id)
            return usdt + btc * ticker.get("last", 0)
        except Exception:
            return 0

    def _get_usdt_balance(self) -> float:
        bal = self.executor.get_balance("USDT")
        return bal.get("USDT", 0.0)

    # ─── 主循环 ────────────────────────────────────
    async def run(self, interval_seconds: int = 3600):
        mode = "DEMO" if self.demo else "LIVE"
        hard_stop_pct = _cfg("hard_stop")
        trail_pct_val = _cfg("trailing_stop")
        daily_dd_pct = _cfg("daily_max_dd")
        cooldown_h = _cfg("cooldown_hours")
        logger.info(f"[{mode}] LiveRunner v2 started on {self.inst_id} {self.bar}")
        logger.info(f"Risk params: hard={hard_stop_pct:.0%} trail={trail_pct_val:.0%} "
                     f"daily={daily_dd_pct:.0%} cooldown={cooldown_h}h")

        # 每日重置计时器
        if self.daily_start_time == 0:
            self.daily_start_time = time.time()
            self.daily_start_equity = self._total_equity()

        while True:
            try:
                # ── 每日重置 ──
                now = time.time()
                if now - self.daily_start_time > 86400:
                    self.daily_start_time = now
                    self.daily_start_equity = self._total_equity()
                    logger.info("[DAILY RESET] new day, equity baseline updated")

                # ── 风控优先 ──
                risk_reason = self._check_risk()
                if risk_reason and risk_reason != "cooldown":
                    self._trigger_risk_response(risk_reason)
                    await asyncio.sleep(interval_seconds)
                    continue
                elif risk_reason == "cooldown":
                    # 冷却中仍记录信号（保持 z-score 连续性），但不交易
                    signal = self._compute_signal()
                    logger.info(f"[COOLDOWN] Signal: {signal:.4f} (not trading)")
                    await asyncio.sleep(interval_seconds)
                    continue

                # ── 信号交易 ──
                signal = self._compute_signal()
                logger.info(f"Signal: {signal:.4f}")

                # 初始保护期：z-score 不足时不做买入
                hist_len = len(self._signal_hist) if hasattr(self, "_signal_hist") else 0

                if signal > ModelConfig.SIGNAL_THRESHOLD and self.position is None:
                    if hist_len < 30:
                        logger.info(f"[GUARD] Signal {signal:.4f} but z-score history={hist_len}<30, skip buy")
                    else:
                        usdt = self._get_usdt_balance()
                        trade_usd = min(usdt, ModelConfig.TRADE_SIZE_USD)
                        if trade_usd < 10:
                            logger.warning("Insufficient balance")
                        else:
                            ticker = OKXExecutor.get_ticker(self.inst_id)
                            price = ticker.get("last", 0)
                            sz = trade_usd / price if price > 0 else 0
                            if sz > 0:
                                ask = ticker.get("ask", price)
                                bal_before = self.executor.get_balance()
                                base_before = bal_before.get(self.base_ccy, 0)
                                oid = self.executor.limit_buy(self.inst_id, sz, ask)
                                if oid:
                                    await asyncio.sleep(3)
                                    bal_after = self.executor.get_balance()
                                    base_after = bal_after.get(self.base_ccy, 0)
                                    filled = base_after - base_before
                                    if filled > 0:
                                        self.position = "long"
                                        self.entry_price = price
                                        self.peak_price = price
                                        logger.success(
                                            f"ENTER LONG: {filled:.6f} {self.base_ccy} "
                                            f"@ ~{price:.1f}  ordId={oid}"
                                        )
                                    else:
                                        logger.warning(f"ORDER NO FILL: ordId={oid}")
                                else:
                                    logger.error(f"ORDER FAILED: buy {sz:.4f} {self.inst_id}")

                elif signal < 0.3 and self.position == "long":
                    self._close_position()
                    self.position = None
                    self.entry_price = 0.0
                    self.peak_price = 0.0
                    logger.success("EXIT LONG (signal)")

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.exception(f"Loop error: {e}")
                await asyncio.sleep(60)

    def _close_position(self):
        bal = self.executor.get_balance()
        amount = bal.get(self.base_ccy, 0)
        if amount > 0:
            ticker = OKXExecutor.get_ticker(self.inst_id)
            bid = ticker.get("bid", ticker.get("last", 0))
            oid = self.executor.limit_sell(self.inst_id, amount, bid)
            if oid:
                logger.success(f"EXIT: sold {amount} {self.base_ccy} @ ~{bid:.1f}  ordId={oid}")
            else:
                logger.error(f"EXIT FAILED: {amount} {self.base_ccy}")


if __name__ == "__main__":
    runner = LiveRunner(demo=True)
    asyncio.run(runner.run())
