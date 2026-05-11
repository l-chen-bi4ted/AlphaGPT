"""
AlphaGPT + OKX 模拟盘运行器 v3 — RiskEngine + MarketRegime 集成版。

新增:
  - RiskEngine: 四级防御 (NORMAL→REDUCED→PROTECTION→EMERGENCY)
  - MarketRegime: ADX/ATR 市场状态感知
  - 环境联动的动态止损/止盈/仓位乘数
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
import pandas as pd
from loguru import logger

from model_core.vm import StackVM
from model_core.config import ModelConfig
from model_core.risk_engine import RiskEngine
from model_core.market_regime import MarketRegime, Regime
from okx_data import fetch_all_candles
from okx_executor import OKXExecutor


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

        # ── 加载公式 ──
        if formula_path is None:
            prefix = f"{self.inst_id.replace('-','')}_{self.bar}"
            formula_path = os.path.join(ModelConfig.SAVE_DIR, f"{prefix}_formula.json")
        with open(formula_path, "r") as f:
            data = json.load(f)
            self.formula = data if isinstance(data, list) else data.get("formula")
        logger.success(f"Loaded formula: {self.formula}")

        self.vm = StackVM()
        self.executor = OKXExecutor(demo=demo)

        # ── 风控引擎 ──
        self.risk = RiskEngine()
        self.regime_detector = MarketRegime()

        # 持仓/信号状态
        self.position = None
        self.last_risk_event = 0.0
        self.daily_start_time = time.time()

        # 启动对账
        self._reconcile()

    # ─── 启动对账 ──────────────────────────────────
    def _reconcile(self):
        """检查实际持仓，登记到 RiskEngine。"""
        try:
            bal = self.executor.get_balance()
            btc = bal.get(self.base_ccy, 0)
            if btc > 0.0001:
                ticker = OKXExecutor.get_ticker(self.inst_id)
                price = ticker.get("last", 0)
                self.position = "long"
                self.risk.add_position(self.inst_id, entry_price=price, amount=btc)
                self.risk.account_state.total_value = self._total_equity()
                logger.info(
                    f"[PROD-V3] Reconciled: {btc:.6f} {self.base_ccy} @ ~{price:.1f} "
                    f"| RiskEngine armed"
                )
        except Exception as e:
            logger.warning(f"Reconcile failed: {e}")

    # ─── 信号计算 ──────────────────────────────────
    def _compute_signal(self, ohlcv: Optional[pd.DataFrame] = None) -> float:
        """拉取 K 线 → 执行公式 → 滚动 z-score → 0~1 信号。"""
        df = ohlcv if ohlcv is not None else fetch_all_candles(self.inst_id, self.bar, limit=200)
        if df.empty or len(df) < 50:
            return 0.5

        device = ModelConfig.DEVICE
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

    # ─── 资产计算 ──────────────────────────────────
    def _total_equity(self) -> float:
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
        logger.info(f"[PROD-V3] LiveRunner started on {self.inst_id} {self.bar}")
        logger.info(f"[PROD-V3] RiskEngine: level={self.risk.risk_level.value}")

        while True:
            try:
                now = time.time()

                # ── 每日重置 ──
                if now - self.daily_start_time > 86400:
                    equity = self._total_equity()
                    self.risk.reset_daily_state(equity)
                    self.daily_start_time = now
                    logger.info(f"[PROD-V3] Daily reset | equity={equity:.0f}")

                # ── 拉数据 + 市场状态 ──
                ohlcv = fetch_all_candles(self.inst_id, self.bar, limit=100)
                if ohlcv is not None and not ohlcv.empty:
                    high_arr = ohlcv["high"].values[-50:].astype(float)
                    low_arr = ohlcv["low"].values[-50:].astype(float)
                    close_arr = ohlcv["close"].values[-50:].astype(float)

                    regime = self.regime_detector.detect(high_arr, low_arr, close_arr)
                    position_mult = self.regime_detector.apply_to_risk_engine(self.risk, regime)

                    ticker = OKXExecutor.get_ticker(self.inst_id)
                    current_price = ticker.get("last", 0)
                else:
                    regime = Regime.TRENDING
                    position_mult = 1.0
                    current_price = 0

                # ── 信号计算 ──
                signal = self._compute_signal(ohlcv)
                hist_len = len(self._signal_hist) if hasattr(self, "_signal_hist") else 0

                logger.info(
                    f"[PROD-V3] regime={regime.value} signal={signal:.4f} "
                    f"price={current_price:.1f} level={self.risk.risk_level.value} mult={position_mult:.1f}"
                )

                # ── 持仓风控（Layer 2）──
                if self.position == "long" and current_price > 0:
                    actions = self.risk.check_position_risk(self.inst_id, current_price)
                    pos = self.risk.positions.get(self.inst_id)
                    actual_pnl = pos.pnl if pos else 0.0

                    if "stop_loss" in actions:
                        logger.critical(f"[PROD-V3] STOP LOSS triggered")
                        self._close_position()
                        self.position = None
                        self.risk.remove_position(self.inst_id)
                        self.risk.update_account_state(actual_pnl)
                        await asyncio.sleep(interval_seconds)
                        continue
                    if "trailing_stop" in actions:
                        logger.critical(f"[PROD-V3] TRAILING STOP triggered")
                        self._close_position()
                        self.position = None
                        self.risk.remove_position(self.inst_id)
                        self.risk.update_account_state(actual_pnl)
                        await asyncio.sleep(interval_seconds)
                        continue
                    if any(a.startswith("take_profit") for a in actions):
                        bal = self.executor.get_balance()
                        amount = bal.get(self.base_ccy, 0) * 0.5
                        if amount > 0 and pos:
                            entry = pos.entry_price
                            sell_pnl = (current_price - entry) * amount
                            ticker = OKXExecutor.get_ticker(self.inst_id)
                            bid = ticker.get("bid", ticker.get("last", 0))
                            oid = self.executor.limit_sell(self.inst_id, amount, bid)
                            if oid:
                                await asyncio.sleep(2)
                                new_bal = self.executor.get_balance()
                                remaining = new_bal.get(self.base_ccy, 0)
                                if pos:
                                    pos.amount = remaining  # 更新 RiskEngine 持仓量
                                logger.success(f"[PROD-V3] TAKE PROFIT 50%: {amount} @ ~{bid:.1f} | remaining={remaining:.6f}")
                                self.risk.update_account_state(sell_pnl)
                        await asyncio.sleep(interval_seconds)
                        continue

                # ── 账户级风控（Layer 3）──
                self.risk.update_daily_pnl(self._total_equity())
                allowed, reason = self.risk.check_account_risk()
                risk_mult = self.risk.get_risk_multiplier()

                if not allowed:
                    logger.critical(f"[PROD-V3] CIRCUIT BREAKER: {reason}")
                    if self.position == "long":
                        self._close_position()
                        self.position = None
                        self.risk.remove_position(self.inst_id)
                        logger.critical("[PROD-V3] All positions closed")
                    await asyncio.sleep(interval_seconds)
                    continue

                # ── 信号交易 ──
                entry_threshold = 0.7 if regime == Regime.RANGING else ModelConfig.SIGNAL_THRESHOLD

                if signal > entry_threshold and self.position is None and risk_mult > 0:
                    if hist_len < 30:
                        logger.info(f"[PROD-V3] GUARD: z-score history={hist_len}<30, skip")
                    else:
                        usdt = self._get_usdt_balance()
                        trade_usd = min(usdt, ModelConfig.TRADE_SIZE_USD) * position_mult * risk_mult
                        if trade_usd < 10:
                            logger.warning("Insufficient balance")
                        else:
                            sz = trade_usd / current_price if current_price > 0 else 0
                            if sz > 0:
                                ask = ticker.get("ask", current_price)
                                oid = self.executor.limit_buy(self.inst_id, sz, ask)
                                if oid:
                                    await asyncio.sleep(3)
                                    bal_after = self.executor.get_balance()
                                    btc_after = bal_after.get(self.base_ccy, 0)
                                    if btc_after > 0:
                                        self.position = "long"
                                        self.risk.add_position(self.inst_id, current_price, btc_after)
                                        logger.success(
                                            f"[PROD-V3] ENTER LONG: {btc_after:.6f} {self.base_ccy} "
                                            f"@ ~{current_price:.1f} mult={position_mult*risk_mult:.1f}"
                                        )

                elif signal < 0.3 and self.position == "long":
                    pos = self.risk.positions.get(self.inst_id)
                    exit_pnl = pos.pnl if pos else 0.0
                    self._close_position()
                    self.position = None
                    self.risk.remove_position(self.inst_id)
                    self.risk.update_account_state(exit_pnl)
                    logger.success(f"[PROD-V3] EXIT LONG (signal) PnL={exit_pnl:+.2f}")

                # 总结
                summary = self.risk.get_summary()
                logger.info(f"[PROD-V3] Risk: {summary}")

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
                logger.success(f"[PROD-V3] EXIT: {amount} {self.base_ccy} @ ~{bid:.1f}")
            else:
                logger.error(f"[PROD-V3] EXIT FAILED: {amount} {self.base_ccy}")


if __name__ == "__main__":
    runner = LiveRunner(demo=True)
    asyncio.run(runner.run())
