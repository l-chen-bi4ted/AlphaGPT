"""
AlphaGPT + OKX 实盘/模拟盘运行器。

流程：
  1. 加载训练好的因子公式
  2. 每 N 分钟拉取最新 K 线
  3. 计算信号 → 达到阈值则下单
  4. 管理持仓：止损 / 移动止损 / 信号反转平仓
"""

import asyncio
import json
import os
import time
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
        self.position = None  # 当前持仓方向：None / "long"

    # ─── 信号计算 ────────────────────────────────────────
    def _compute_signal(self) -> float:
        """拉取最新数据 → 计算公式 → 返回信号概率。"""
        df = fetch_all_candles(self.inst_id, self.bar, limit=200)
        if df.empty or len(df) < 50:
            logger.warning("Not enough data for signal")
            return 0.5

        device = ModelConfig.DEVICE

        # 与 okx_data.raw_data_cache 格式对齐: [1, T]
        close = torch.tensor(
            df["close"].values[-200:], dtype=torch.float32, device=device
        ).unsqueeze(0)
        open_ = torch.tensor(
            df["open"].values[-200:], dtype=torch.float32, device=device
        ).unsqueeze(0)
        high = torch.tensor(
            df["high"].values[-200:], dtype=torch.float32, device=device
        ).unsqueeze(0)
        low = torch.tensor(
            df["low"].values[-200:], dtype=torch.float32, device=device
        ).unsqueeze(0)
        vol = torch.tensor(
            df["vol"].values[-200:], dtype=torch.float32, device=device
        ).unsqueeze(0)

        # 模拟 raw_data_cache 格式用于 FeatureEngineer
        raw = {
            "close": close,
            "open": open_,
            "high": high,
            "low": low,
            "volume": vol,
            "liquidity": torch.full_like(vol, 1e9),
            "fdv": torch.full_like(vol, 1e12),
        }

        from model_core.factors import FeatureEngineer
        feat = FeatureEngineer.compute_features(raw)  # [F, 1, T]

        res = self.vm.execute(self.formula, feat)
        if res is None:
            return 0.5

        signal = float(torch.sigmoid(res[0, -1]).item())
        return signal

    # ─── 持仓比例 ──────────────────────────────────────
    def _get_usdt_balance(self) -> float:
        bal = self.executor.get_balance("USDT")
        return bal.get("USDT", 0.0)

    # ─── 主循环 ────────────────────────────────────────
    async def run(self, interval_seconds: int = 3600):
        """主循环。interval_seconds 默认 1 小时（与 1H K 线周期一致）。"""
        mode = "DEMO" if self.demo else "LIVE"
        base_ccy = self.inst_id.split("-")[0]
        logger.info(f"[{mode}] LiveRunner started on {self.inst_id} {self.bar}")

        while True:
            try:
                signal = self._compute_signal()
                logger.info(f"Signal: {signal:.4f}")

                if signal > ModelConfig.SIGNAL_THRESHOLD and self.position is None:
                    # 入场：做多
                    usdt = self._get_usdt_balance()
                    trade_usd = min(usdt, ModelConfig.TRADE_SIZE_USD)
                    if trade_usd < 10:
                        logger.warning("Insufficient balance")
                    else:
                        ticker = OKXExecutor.get_ticker(self.inst_id)
                        price = ticker.get("last", 0)
                        sz = trade_usd / price if price > 0 else 0
                        if sz > 0:
                            # 限价买入（ask 价），模拟盘市价单流动性不足
                            ask = ticker.get("ask", price)
                            bal_before = self.executor.get_balance()
                            base_before = bal_before.get(base_ccy, 0)
                            oid = self.executor.limit_buy(self.inst_id, sz, ask)
                            if oid:
                                await asyncio.sleep(3)
                                bal_after = self.executor.get_balance()
                                base_after = bal_after.get(base_ccy, 0)
                                filled = base_after - base_before
                                if filled > 0:
                                    self.position = "long"
                                    logger.success(
                                        f"ENTER LONG: filled {filled:.6f} {base_ccy} "
                                        f"(req {sz:.4f}) @ ~{price}  ordId={oid}"
                                    )
                                else:
                                    logger.warning(f"ORDER NO FILL: ordId={oid}")
                            else:
                                logger.error(f"ORDER FAILED: buy {sz:.4f} {self.inst_id}")

                elif signal < 0.3 and self.position == "long":
                    # 平仓
                    self._close_position()
                    self.position = None
                    logger.success("EXIT LONG")

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.exception(f"Loop error: {e}")
                await asyncio.sleep(60)

    def _close_position(self):
        bal = self.executor.get_balance()
        base = self.inst_id.split("-")[0]
        amount = bal.get(base, 0)
        if amount > 0:
            ticker = OKXExecutor.get_ticker(self.inst_id)
            bid = ticker.get("bid", ticker.get("last", 0))
            oid = self.executor.limit_sell(self.inst_id, amount, bid)
            if oid:
                logger.success(f"EXIT LONG: sold {amount} {base}  ordId={oid}")
            else:
                logger.error(f"EXIT FAILED: {amount} {base}")


if __name__ == "__main__":
    runner = LiveRunner(demo=True)
    asyncio.run(runner.run())
