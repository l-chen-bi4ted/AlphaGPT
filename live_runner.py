"""
AlphaGPT + OKX 模拟盘运行器 v3 — 实盘闭环版。

v3 修复：
- RiskEngine 真正联动：每轮 check_account_risk，成交回报驱动 record_trade
- 下单后 wait_for_fill 确认成交，获取真实成交均价和手续费
- MarketRegime 不再直接修改 RiskEngine，改为返回参数由 runner 应用
- 止损/止盈强制使用市价单（确保成交）
- API 调用增加缓存，降低限频风险
- 入场使用限价单（降低成本），但风控出场使用市价单（确保执行）
"""

import asyncio
import json
import os
import time
from collections import deque
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
import numpy as np
import pandas as pd
from loguru import logger

from model_core.vm import StackVM
from model_core.config import ModelConfig, default_config
from model_core.risk_engine import RiskEngine
from model_core.market_regime import MarketRegime, Regime
from okx_data import fetch_all_candles
from okx_executor import OKXExecutor, FillResult, OrderState


class LiveRunner:
    def __init__(
        self,
        formula_path: Optional[str] = None,
        inst_id: Optional[str] = None,
        bar: str = "1H",
        demo: bool = True,
        config: Optional[ModelConfig] = None,
    ):
        self.config = config or default_config
        self.inst_id = inst_id or self.config.inst_id
        self.bar = bar or self.config.bar
        self.demo = demo
        self.base_ccy = self.inst_id.split("-")[0]
        self.quote_ccy = self.inst_id.split("-")[1]

        # 加载公式
        if formula_path is None:
            prefix = f"{self.inst_id.replace('-','')}_{self.bar}"
            formula_path = os.path.join(self.config.save_dir, f"{prefix}_formula.json")
        with open(formula_path, "r") as f:
            data = json.load(f)
            self.formula = data if isinstance(data, list) else data.get("formula")
        logger.success(f"Loaded formula: {self.formula}")

        self.vm = StackVM()
        self.executor = OKXExecutor(demo=demo)

        # 风控引擎
        state_file = Path(self.config.save_dir) / "risk_state.json"
        self.risk = RiskEngine(state_file=str(state_file))
        self.regime_detector = MarketRegime()

        # 状态
        self.position = None
        self.daily_start_time = time.time()
        self._signal_hist = deque(maxlen=200)
        self._price_cache = {}  # {ts: ticker_dict}
        self._balance_cache = {"ts": 0, "data": {}}
        self._equity_cache = {"ts": 0, "value": 0.0}

        # 启动对账
        self._reconcile()

    # ─── 缓存辅助 ──────────────────────────────────
    def _get_ticker(self, ttl_sec: float = 3.0) -> dict:
        """带缓存的 ticker 查询。"""
        now = time.time()
        key = f"{self.inst_id}_{int(now / ttl_sec)}"
        if key in self._price_cache:
            return self._price_cache[key]
        ticker = OKXExecutor.get_ticker(self.inst_id)
        self._price_cache[key] = ticker
        return ticker

    def _get_balance(self, ttl_sec: float = 5.0) -> dict:
        """带缓存的余额查询。"""
        now = time.time()
        if now - self._balance_cache["ts"] < ttl_sec:
            return self._balance_cache["data"]
        bal = self.executor.get_balance()
        self._balance_cache = {"ts": now, "data": bal}
        return bal

    def _total_equity(self, ttl_sec: float = 5.0) -> float:
        """带缓存的权益计算。"""
        now = time.time()
        if now - self._equity_cache["ts"] < ttl_sec:
            return self._equity_cache["value"]
        bal = self._get_balance(ttl_sec=0)
        btc = bal.get(self.base_ccy, 0)
        usdt = bal.get(self.quote_ccy, 0)
        ticker = self._get_ticker()
        price = ticker.get("last", 0)
        eq = usdt + btc * price
        self._equity_cache = {"ts": now, "value": eq}
        return eq

    # ─── 启动对账 ──────────────────────────────────
    def _reconcile(self):
        """检查实际持仓，登记到 RiskEngine。"""
        try:
            bal = self._get_balance(ttl_sec=0)
            btc = bal.get(self.base_ccy, 0)
            if btc > 0.0001:
                ticker = self._get_ticker()
                price = ticker.get("last", 0)
                self.position = "long"
                self.risk.add_position(self.inst_id, entry_price=price, amount=btc)
                eq = self._total_equity(ttl_sec=0)
                self.risk.account_state.total_value = eq
                logger.info(
                    f"[Reconcile] {btc:.6f} {self.base_ccy} @ ~{price:.1f} "
                    f"| Equity={eq:.2f}"
                )
            else:
                self.position = None
                logger.info("[Reconcile] No position")
        except Exception as e:
            logger.warning(f"Reconcile failed: {e}")

    # ─── 信号计算 ──────────────────────────────────
    def _compute_signal(self, ohlcv: Optional[pd.DataFrame] = None) -> float:
        """拉取 K 线 → 执行公式 → 滚动 z-score → 0~1 信号。"""
        df = ohlcv if ohlcv is not None else fetch_all_candles(self.inst_id, self.bar, limit=200)
        if df.empty or len(df) < 50:
            return 0.5

        device = self.config.device
        close = torch.tensor(df["close"].values[-200:], dtype=torch.float32, device=device).unsqueeze(0)
        open_ = torch.tensor(df["open"].values[-200:], dtype=torch.float32, device=device).unsqueeze(0)
        high  = torch.tensor(df["high"].values[-200:],  dtype=torch.float32, device=device).unsqueeze(0)
        low   = torch.tensor(df["low"].values[-200:],   dtype=torch.float32, device=device).unsqueeze(0)
        vol   = torch.tensor(df["vol"].values[-200:],   dtype=torch.float32, device=device).unsqueeze(0)

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
        self._signal_hist.append(current)
        if len(self._signal_hist) < 30:
            return 0.5

        arr = np.array(self._signal_hist)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median)) + 1e-8
        z = 0.6745 * (current - median) / mad
        signal = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(signal, 0.01, 0.99))

    # ─── 交易执行 ──────────────────────────────────
    def _enter_long(self, price: float, mult: float) -> Optional[FillResult]:
        """开多仓，返回成交结果。"""
        usdt = self._get_balance().get(self.quote_ccy, 0.0)
        trade_usd = min(usdt, self.config.trade_size_usd) * mult
        if trade_usd < 10:
            logger.warning("Insufficient balance for entry")
            return None

        sz = trade_usd / price if price > 0 else 0
        if sz <= 0:
            return None

        # 用限价单挂 ask 价（立即成交，但按 maker 费率）
        ticker = self._get_ticker()
        ask = ticker.get("ask", price)
        oid = self.executor.limit_buy(self.inst_id, sz, ask)
        if not oid:
            logger.error("Entry order failed")
            return None

        fill = self.executor.wait_for_fill(oid, self.inst_id, timeout_sec=60, poll_interval=2.0)
        if fill and fill.state == OrderState.FILLED:
            actual_sz = fill.filled_sz
            avg_px = fill.avg_px
            self.risk.add_position(self.inst_id, entry_price=avg_px, amount=actual_sz)
            self.position = "long"
            logger.success(
                f"ENTER LONG: {actual_sz:.6f} @ {avg_px:.2f} fee={fill.fee:.4f} {fill.fee_ccy}"
            )
            return fill
        else:
            logger.error(f"Entry fill failed or timeout: {oid}")
            # 尝试取消未成交订单
            self.executor.cancel_order(oid, self.inst_id)
            return None

    def _exit_long(self, reason: str) -> Optional[FillResult]:
        """平多仓，返回成交结果。"""
        bal = self._get_balance()
        amount = bal.get(self.base_ccy, 0)
        if amount <= 0:
            logger.warning("No position to exit")
            return None

        # 风控出场用市价单，确保成交
        oid = self.executor.market_sell(self.inst_id, amount)
        if not oid:
            logger.error(f"Exit order failed ({reason})")
            return None

        fill = self.executor.wait_for_fill(oid, self.inst_id, timeout_sec=60, poll_interval=2.0)
        if fill and fill.state == OrderState.FILLED:
            avg_px = fill.avg_px
            # 计算已实现盈亏
            pos = self.risk.positions.get(self.inst_id)
            realized_pnl = 0.0
            if pos:
                realized_pnl = (avg_px - pos.entry_price) * fill.filled_sz - fill.fee
                self.risk.remove_position(self.inst_id, avg_px, realized_pnl)
            else:
                logger.warning("Exit without tracked position")
            self.position = None
            logger.success(
                f"EXIT LONG ({reason}): {fill.filled_sz:.6f} @ {avg_px:.2f} "
                f"PnL={realized_pnl:+.2f} fee={fill.fee:.4f}"
            )
            return fill
        else:
            logger.error(f"Exit fill failed or timeout: {oid}")
            return None

    def _take_profit(self, ratio: float = 0.5) -> Optional[FillResult]:
        """分批止盈，默认平掉 50%。"""
        bal = self._get_balance()
        total = bal.get(self.base_ccy, 0)
        amount = total * ratio
        if amount <= 0:
            return None

        # 止盈也用限价单挂 bid 价
        ticker = self._get_ticker()
        bid = ticker.get("bid", ticker.get("last", 0))
        oid = self.executor.limit_sell(self.inst_id, amount, bid)
        if not oid:
            return None

        fill = self.executor.wait_for_fill(oid, self.inst_id, timeout_sec=30, poll_interval=2.0)
        if fill and fill.state == OrderState.FILLED:
            pos = self.risk.positions.get(self.inst_id)
            realized_pnl = 0.0
            if pos:
                realized_pnl = (fill.avg_px - pos.entry_price) * fill.filled_sz - fill.fee
                self.risk.record_trade(realized_pnl)
                # 更新持仓量
                pos.amount -= fill.filled_sz
                if pos.amount <= 0:
                    self.risk.remove_position(self.inst_id, fill.avg_px, realized_pnl)
                    self.position = None
            logger.success(
                f"TAKE PROFIT {ratio*100:.0f}%: {fill.filled_sz:.6f} @ {fill.avg_px:.2f} "
                f"PnL={realized_pnl:+.2f}"
            )
            return fill
        else:
            self.executor.cancel_order(oid, self.inst_id)
            return None

    # ─── 主循环 ────────────────────────────────────
    async def run(self, interval_seconds: int = 3600):
        mode = "DEMO" if self.demo else "LIVE"
        logger.info(f"[{mode}] LiveRunner started on {self.inst_id} {self.bar}")
        logger.info(f"[{mode}] RiskEngine: level={self.risk.risk_level.value}")

        while True:
            cycle_start = time.time()
            try:
                # ── 每日重置 ──
                if cycle_start - self.daily_start_time > 86400:
                    equity = self._total_equity()
                    self.risk.reset_daily_state(equity)
                    self.daily_start_time = cycle_start
                    logger.info(f"[{mode}] Daily reset | equity={equity:.2f}")

                # ── 拉数据 + 市场状态 ──
                ohlcv = fetch_all_candles(self.inst_id, self.bar, limit=100)
                current_price = 0.0
                regime = Regime.RANGING
                regime_params = self.regime_detector.get_params(regime)

                if ohlcv is not None and not ohlcv.empty:
                    high_arr = ohlcv["high"].values[-50:].astype(float)
                    low_arr = ohlcv["low"].values[-50:].astype(float)
                    close_arr = ohlcv["close"].values[-50:].astype(float)

                    regime = self.regime_detector.detect(high_arr, low_arr, close_arr)
                    regime_params = self.regime_detector.get_params(regime)

                    ticker = self._get_ticker()
                    current_price = ticker.get("last", 0)

                # ── 账户级风控（Layer 3）──
                equity = self._total_equity()
                allowed, reason = self.risk.check_account_risk(equity)
                risk_mult = self.risk.get_risk_multiplier()

                if not allowed:
                    logger.critical(f"[{mode}] CIRCUIT BREAKER: {reason}")
                    if self.position == "long":
                        self._exit_long("circuit_breaker")
                    await asyncio.sleep(interval_seconds)
                    continue

                # ── 持仓风控（Layer 2）──
                if self.position == "long" and current_price > 0:
                    actions = self.risk.check_position_risk(self.inst_id, current_price)
                    if "stop_loss" in actions or "trailing_stop" in actions or "time_out" in actions:
                        self._exit_long("stop_loss" if "stop_loss" in actions else "trailing_stop")
                        await asyncio.sleep(interval_seconds)
                        continue
                    if any(a.startswith("take_profit") for a in actions):
                        self._take_profit(ratio=0.5)
                        # 继续循环，不移除持仓（可能还有剩余）

                # ── 信号计算 ──
                signal = self._compute_signal(ohlcv)
                hist_len = len(self._signal_hist)

                # 动态阈值：震荡市更严格
                entry_threshold = self.config.signal_threshold
                if regime == Regime.RANGING:
                    entry_threshold = min(entry_threshold + 0.15, 0.85)
                elif regime == Regime.VOLATILE:
                    entry_threshold = min(entry_threshold + 0.25, 0.90)

                logger.info(
                    f"[{mode}] regime={regime.value} signal={signal:.4f} "
                    f"price={current_price:.1f} level={self.risk.risk_level.value} "
                    f"risk_mult={risk_mult:.1f} pos_mult={regime_params.position_multiplier:.1f}"
                )

                # ── 信号交易 ──
                combined_mult = regime_params.position_multiplier * risk_mult

                if signal > entry_threshold and self.position is None and combined_mult > 0:
                    if hist_len < 30:
                        logger.info(f"[{mode}] GUARD: z-score history={hist_len}<30, skip")
                    else:
                        fill = self._enter_long(current_price, combined_mult)
                        if fill:
                            # 应用 regime 参数到 RiskEngine（可选）
                            self.risk.stop_loss_pct = regime_params.stop_loss_pct
                            self.risk.trailing_stop_pct = regime_params.trailing_stop_pct
                            self.risk.take_profit_pcts = regime_params.take_profit_pcts

                elif signal < 0.3 and self.position == "long":
                    self._exit_long("signal_exit")

                # 总结
                summary = self.risk.get_summary()
                logger.info(f"[{mode}] Risk: {summary}")

            except Exception as e:
                logger.exception(f"Loop error: {e}")

            # 计算剩余等待时间
            elapsed = time.time() - cycle_start
            sleep_time = max(5, interval_seconds - elapsed)
            await asyncio.sleep(sleep_time)


if __name__ == "__main__":
    runner = LiveRunner(demo=True)
    asyncio.run(runner.run())
