"""
OKX Trade Executor v3 — 实盘/模拟盘下单模块（成交闭环版）。

v3 修复：
- 下单后支持查询成交状态（get_order / wait_for_fill）
- 统一使用 loguru 日志
- 查询实际手续费和成交均价
- 异常分类处理，不吞掉所有错误
"""

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional, Dict
from dataclasses import dataclass
from enum import Enum

import requests
from loguru import logger


OKX_REST_URL = "https://www.okx.com"
PROXY_NONE = {"http": None, "https": None}


class OrderState(Enum):
    LIVE = "live"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass
class FillResult:
    """成交结果。"""
    ord_id: str
    state: OrderState
    filled_sz: float
    avg_px: float
    fee: float
    fee_ccy: str
    pnl: float = 0.0
    realized_pnl: float = 0.0
    raw: dict = None


class OKXExecutor:
    """OKX 现货交易执行器 v3。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        passphrase: Optional[str] = None,
        demo: bool = True,
    ):
        self.demo = demo
        prefix = "OKX_DEMO_" if demo else "OKX_"
        self.api_key = api_key or os.getenv(f"{prefix}API_KEY", "")
        self.secret_key = secret_key or os.getenv(f"{prefix}SECRET_KEY", "")
        self.passphrase = passphrase or os.getenv(f"{prefix}PASSPHRASE", "")

        if not all([self.api_key, self.secret_key, self.passphrase]):
            logger.warning("OKX credentials incomplete — trading will fail")

    # ─── 签名 ────────────────────────────────────────────
    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        payload = f"{timestamp}{method.upper()}{path}{body}"
        sig = hmac.new(
            self.secret_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(sig).decode("utf-8")

    @staticmethod
    def _iso_timestamp() -> str:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

    def _headers(self, method: str, path: str, body: str = "") -> dict:
        ts = self._iso_timestamp()
        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": self._sign(ts, method, path, body),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 Chrome/120.0",
        }
        if self.demo:
            headers["x-simulated-trading"] = "1"
        return headers

    # ─── 请求 ────────────────────────────────────────────
    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        if method == "GET" and body:
            from urllib.parse import urlencode
            qs = urlencode(body)
            path = f"{path}?{qs}"
            body_str = ""
            params = None
        else:
            body_str = json.dumps(body) if body else ""
            params = body if method == "GET" else None
        url = f"{OKX_REST_URL}{path}"
        headers = self._headers(method, path, body_str)

        for attempt in range(3):
            try:
                if method == "GET":
                    resp = requests.get(
                        url, headers=headers, timeout=10, params=params, proxies=PROXY_NONE
                    )
                else:
                    resp = requests.post(
                        url, headers=headers, data=body_str, timeout=10, proxies=PROXY_NONE
                    )

                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                if data.get("code") != "0":
                    logger.error(f"OKX error: {data.get('msg', 'unknown')} (code={data.get('code')})")
                    return {"error": data.get("msg", "unknown"), "code": data.get("code")}
                return data
            except requests.RequestException as e:
                logger.error(f"Request error (attempt {attempt+1}): {e}")
                time.sleep(1)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error (attempt {attempt+1}): {e}")
                time.sleep(1)

        return {"error": "max retries"}

    # ─── 下单 ────────────────────────────────────────────
    def _place_order(
        self,
        ord_type: str,
        side: str,
        inst_id: str,
        sz: float,
        px: Optional[float] = None,
    ) -> Optional[str]:
        body = {
            "instId": inst_id,
            "tdMode": "cash",
            "side": side,
            "ordType": ord_type,
            "sz": str(sz),
        }
        if px is not None:
            body["px"] = str(px)

        data = self._request("POST", "/api/v5/trade/order", body)
        if "error" in data:
            return None

        ord_id = (data.get("data") or [{}])[0].get("ordId")
        if ord_id:
            logger.info(f"✓ {side.upper()} {sz} {inst_id} → ordId={ord_id}")
        return ord_id

    def market_buy(self, inst_id: str, sz: float) -> Optional[str]:
        return self._place_order("market", "buy", inst_id, sz)

    def market_sell(self, inst_id: str, sz: float) -> Optional[str]:
        return self._place_order("market", "sell", inst_id, sz)

    def limit_buy(self, inst_id: str, sz: float, px: float) -> Optional[str]:
        return self._place_order("limit", "buy", inst_id, sz, px)

    def limit_sell(self, inst_id: str, sz: float, px: float) -> Optional[str]:
        return self._place_order("limit", "sell", inst_id, sz, px)

    def cancel_order(self, ord_id: str, inst_id: str) -> bool:
        data = self._request("POST", "/api/v5/trade/cancel-order", {
            "instId": inst_id,
            "ordId": ord_id,
        })
        return "error" not in data

    # ─── 成交回报查询 ────────────────────────────────────
    def get_order(self, ord_id: str, inst_id: str) -> Optional[FillResult]:
        """
        查询订单成交详情。
        返回 FillResult 或 None（查询失败）。
        """
        data = self._request("GET", "/api/v5/trade/order", {
            "instId": inst_id,
            "ordId": ord_id,
        })
        if "error" in data:
            return None

        info = (data.get("data") or [{}])[0]
        state_str = info.get("state", "")
        state_map = {
            "live": OrderState.LIVE,
            "partially_filled": OrderState.PARTIALLY_FILLED,
            "filled": OrderState.FILLED,
            "canceled": OrderState.CANCELED,
            "reject": OrderState.REJECTED,
        }

        return FillResult(
            ord_id=ord_id,
            state=state_map.get(state_str, OrderState.UNKNOWN),
            filled_sz=float(info.get("accFillSz", 0) or 0),
            avg_px=float(info.get("avgPx", 0) or 0),
            fee=float(info.get("fee", 0) or 0),
            fee_ccy=info.get("feeCcy", ""),
            pnl=float(info.get("pnl", 0) or 0),
            realized_pnl=float(info.get("realizedPnl", 0) or 0),
            raw=info,
        )

    def wait_for_fill(
        self,
        ord_id: str,
        inst_id: str,
        timeout_sec: int = 60,
        poll_interval: float = 2.0,
    ) -> Optional[FillResult]:
        """
        轮询等待订单完全成交。
        返回 FillResult 或 None（超时/失败）。
        """
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            fill = self.get_order(ord_id, inst_id)
            if fill is None:
                logger.warning(f"Order query failed for {ord_id}, retrying...")
                time.sleep(poll_interval)
                continue

            if fill.state == OrderState.FILLED:
                logger.info(
                    f"✓ Order {ord_id} filled: sz={fill.filled_sz} "
                    f"avg_px={fill.avg_px} fee={fill.fee} {fill.fee_ccy}"
                )
                return fill
            elif fill.state in (OrderState.CANCELED, OrderState.REJECTED):
                logger.error(f"✗ Order {ord_id} {fill.state.value}")
                return fill
            else:
                logger.debug(f"Order {ord_id} state: {fill.state.value}, waiting...")

            time.sleep(poll_interval)

        logger.error(f"Order {ord_id} fill timeout after {timeout_sec}s")
        return None

    # ─── 余额 ────────────────────────────────────────────
    def get_balance(self, ccy: Optional[str] = None) -> Dict[str, float]:
        params = {}
        if ccy:
            params["ccy"] = ccy
        data = self._request("GET", "/api/v5/account/balance", params)
        if "error" in data:
            return {}

        balances = {}
        for row in data.get("data", []):
            for detail in row.get("details", []):
                c = detail["ccy"]
                balances[c] = float(detail.get("availBal", 0))
        return balances

    # ─── 持仓 ────────────────────────────────────────────
    def get_positions(self, inst_id: Optional[str] = None) -> list[dict]:
        data = self._request("GET", "/api/v5/account/positions", {})
        if "error" in data:
            return []
        positions = data.get("data", [])
        if inst_id:
            positions = [p for p in positions if p.get("instId") == inst_id]
        return positions

    # ─── 行情（公开 API）─────────────────────────────────
    @staticmethod
    def get_ticker(inst_id: str) -> dict:
        try:
            resp = requests.get(
                f"{OKX_REST_URL}/api/v5/market/ticker",
                params={"instId": inst_id},
                headers={"User-Agent": "Mozilla/5.0 Chrome/120.0"},
                timeout=10,
                proxies=PROXY_NONE,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "0":
                return {}
            ticker = (data.get("data") or [{}])[0]
            return {
                "last": float(ticker.get("last", 0)),
                "bid": float(ticker.get("bidPx", 0)),
                "ask": float(ticker.get("askPx", 0)),
                "vol_24h": float(ticker.get("vol24h", 0)),
            }
        except Exception as e:
            logger.error(f"Ticker fetch error: {e}")
            return {}


# ─── 快速测试 ──────────────────────────────────────────────
if __name__ == "__main__":
    exec = OKXExecutor(demo=True)
    logger.info(f"Balances: {exec.get_balance()}")
    ticker = OKXExecutor.get_ticker("BTC-USDT")
    logger.info(f"BTC-USDT: {ticker}")
