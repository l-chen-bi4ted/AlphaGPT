"""
OKX Trade Executor — 实盘/模拟盘下单模块。

通过 OKX REST API（HMAC-SHA256 签名）执行现货交易。
模拟盘通过 X-Simulated-Trading 头启用，无需真实资金。

用法：
    exec = OKXExecutor(api_key="...", secret="...", passphrase="...", demo=True)
    order_id = exec.market_buy("BTC-USDT", 0.01)      # 市价买入 0.01 BTC
    order_id = exec.market_sell("BTC-USDT", 0.01)     # 市价卖出 0.01 BTC
    balance  = exec.get_balance("USDT")                # 查余额
"""

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

import requests


OKX_REST_URL = "https://www.okx.com"


class OKXExecutor:
    """OKX 现货交易执行器。"""

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
            # GET query params must be in the signed path
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
                    resp = requests.get(url, headers=headers, timeout=10, params=params)
                else:
                    resp = requests.post(url, headers=headers, data=body_str, timeout=10)

                if resp.status_code == 429:
                    wait = 2 ** attempt
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                data = resp.json()
                if data.get("code") != "0":
                    print(f"  OKX error: {data.get('msg', 'unknown')} (code={data.get('code')})")
                    return {"error": data.get("msg", "unknown"), "code": data.get("code")}
                return data
            except Exception as e:
                print(f"  Request error (attempt {attempt+1}): {e}")
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
        """
        下单（市价/限价）。
        返回 order_id 或 None。
        """
        body = {
            "instId": inst_id,
            "tdMode": "cash",      # 现货现金模式
            "side": side,          # buy / sell
            "ordType": ord_type,   # market / limit
            "sz": str(sz),
        }
        if px is not None:
            body["px"] = str(px)

        data = self._request("POST", "/api/v5/trade/order", body)
        if "error" in data:
            return None

        ord_id = (data.get("data") or [{}])[0].get("ordId")
        if ord_id:
            print(f"  ✓ {side.upper()} {sz} {inst_id} → ordId={ord_id}")
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

    # ─── 余额 ────────────────────────────────────────────
    def get_balance(self, ccy: Optional[str] = None) -> dict[str, float]:
        """
        获取可用余额。ccy 为空则返回所有币种。
        返回 {ccy: availBal}。
        """
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
        """获取现货持仓（实际上就是各币种余额列表）。"""
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
        resp = requests.get(
            f"{OKX_REST_URL}/api/v5/market/ticker",
            params={"instId": inst_id},
            headers={"User-Agent": "Mozilla/5.0 Chrome/120.0"},
            timeout=10,
        )
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


# ─── 快速测试 ──────────────────────────────────────────────
if __name__ == "__main__":
    exec = OKXExecutor(demo=True)
    print("Balances:", exec.get_balance())
    ticker = OKXExecutor.get_ticker("BTC-USDT")
    print(f"BTC-USDT: {ticker}")
