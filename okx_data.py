"""
OKX Market Data Connector — 替代 PostgreSQL/CryptoDataLoader。

从 OKX REST API 拉取 K 线数据，输出兼容 FeatureEngineer 的 raw_data_cache。
单标的模式：[1, T] 张量；多标的需要将 shape 从 [N, T] 传递给 FeatureEngineer。

OKX 公开行情 API 无需鉴权，限频 20 req / 2s。
"""

import time
import json
import os
from typing import Optional

import numpy as np
import torch
import requests
import pandas as pd

from model_core.config import ModelConfig

# ─── OKX API 常量 ───────────────────────────────────────────
OKX_REST_URL = "https://www.okx.com"
CANDLES_PATH = "/api/v5/market/candles"           # 近 2 天
HISTORY_CANDLES_PATH = "/api/v5/market/history-candles"  # 历史（> 2 天前）

BAR_MAP = {
    "1m":  "1m",  "3m": "3m",  "5m": "5m",  "15m": "15m",
    "30m": "30m", "1H": "1H",  "2H": "2H",  "4H": "4H",
    "6H":  "6H",  "12H": "12H", "1D": "1D",
}


def _parse_candle(row: list) -> dict:
    """将 OKX 单根 K 线数据转为 float dict。"""
    return {
        "ts": int(row[0]),
        "open":  float(row[1]),
        "high":  float(row[2]),
        "low":   float(row[3]),
        "close": float(row[4]),
        "vol":   float(row[5]),       # 成交量（张/币数）
        "vol_ccy": float(row[6]) if len(row) > 6 and row[6] else float(row[5]) * float(row[4]),
    }


def _fetch_candles_page(
    inst_id: str,
    bar: str = "1H",
    after: Optional[str] = None,
    before: Optional[str] = None,
    limit: int = 100,
    use_history: bool = False,
) -> list[dict]:
    """单页 K 线请求（带重试）。"""
    path = HISTORY_CANDLES_PATH if use_history else CANDLES_PATH
    params = {"instId": inst_id, "bar": BAR_MAP.get(bar, bar), "limit": min(limit, 300)}
    if after:
        params["after"] = str(after)
    if before:
        params["before"] = str(before)

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    for attempt in range(3):
        try:
            resp = requests.get(
                f"{OKX_REST_URL}{path}", params=params, timeout=15, headers=headers
            )
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            body = resp.json()
            if body.get("code") != "0":
                print(f"  OKX error: {body.get('msg', 'unknown')}")
                return []
            return [_parse_candle(row) for row in body.get("data", [])]
        except Exception as e:
            print(f"  Fetch error (attempt {attempt+1}): {e}")
            time.sleep(1)
    return []


def fetch_all_candles(
    inst_id: str,
    bar: str = "1H",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    拉取 instId 的全部历史 K 线，返回 DataFrame (ts, o, h, l, c, vol, volCcy)。
    
    内部自动分页，先拉近 2 天数据，再补历史。
    """
    all_rows = []

    # 先拉历史（after=0 表示从最早开始，向后翻页）
    after = "0"
    while len(all_rows) < limit:
        page = _fetch_candles_page(inst_id, bar, after=after, limit=300, use_history=True)
        if not page:
            break
        all_rows.extend(page)
        if len(page) < 300:
            break
        after = str(page[-1]["ts"])
        time.sleep(0.1)  # 温和限频

    # 再拉近 2 天数据（可能重叠，后续去重）
    recent = _fetch_candles_page(inst_id, bar, limit=300, use_history=False)
    if recent:
        all_rows.extend(recent)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    if limit:
        df = df.tail(limit)
    return df


class OKXDataLoader:
    """
    OKX 数据加载器，输出格式兼容 FeatureEngineer.compute_features。

    单标的模式：
        raw_data_cache = {'open': [1, T], 'high': [1, T], ...}
        feat_tensor     = [F, 1, T]
        target_ret      = [1, T]
    """

    def __init__(
        self,
        inst_id: str = "BTC-USDT",
        bar: str = "1H",
        limit: int = 2000,
        train_ratio: float = 0.7,
        cache_dir: Optional[str] = None,
    ):
        self.inst_id = inst_id
        self.bar = bar
        self.limit = limit
        self.train_ratio = train_ratio
        # 默认缓存目录：项目根 data_cache/
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "data_cache")
        self.cache_dir = cache_dir

        self.feat_tensor: Optional[torch.Tensor] = None
        self.raw_data_cache: Optional[dict] = None
        self.target_ret: Optional[torch.Tensor] = None
        self.full_df: Optional[pd.DataFrame] = None  # 原始数据（用于回测可视化）

    def _cache_path(self) -> str:
        """本地缓存文件路径：data_cache/BTCUSDT_1H.csv"""
        fname = f"{self.inst_id.replace('-', '')}_{self.bar}.csv"
        return os.path.join(self.cache_dir, fname)

    def load_data(self):
        """优先读本地 CSV 缓存；无缓存则联网拉取。"""
        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            print(f"Loading from cache: {cache_path}")
            df = pd.read_csv(cache_path).tail(self.limit)
        else:
            print(f"Fetching {self.inst_id} {self.bar} candles from OKX...")
            df = fetch_all_candles(self.inst_id, self.bar, self.limit)
            if df.empty:
                raise RuntimeError(f"No data for {self.inst_id}")

        print(f"  Got {len(df)} candles: {df['ts'].min()} → {df['ts'].max()}")
        self.full_df = df

        # 转为张量 [1, T]
        device = ModelConfig.DEVICE
        t = lambda col: torch.tensor(df[col].values, dtype=torch.float32, device=device).unsqueeze(0)

        # 用成交量近似流动性（CEX 流动性充足）；FDV 用大常数
        vol_usd = torch.tensor(
            df["vol_ccy"].values, dtype=torch.float32, device=device
        ).unsqueeze(0)

        self.raw_data_cache = {
            "open":      t("open"),
            "high":      t("high"),
            "low":       t("low"),
            "close":     t("close"),
            "volume":    t("vol"),
            "liquidity": vol_usd,               # USD 量 ≈ 流动性
            "fdv":       torch.full_like(vol_usd, 1e12),  # 大常数，liq_score ≈ 1
        }

        # 特征张量 [F, 1, T]
        from model_core.factors import FeatureEngineer
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)

        # 目标：下期收益率（t 时刻因子预测 t→t+1 收益，对齐实盘推理）
        close = self.raw_data_cache["close"]  # [1, T]
        t1 = torch.roll(close, -1, dims=1)
        self.target_ret = torch.log(t1 / (close + 1e-9))
        self.target_ret[:, -1:] = 0.0

        print(f"Data ready. Shape: {self.feat_tensor.shape}")

    def train_test_split(self):
        """按时间切分训练/测试集。"""
        T = self.feat_tensor.shape[-1]
        split = int(T * self.train_ratio)
        return {
            "feat": self.feat_tensor[:, :, :split],
            "raw": {k: v[:, :split] for k, v in self.raw_data_cache.items()},
            "target": self.target_ret[:, :split],
        }, {
            "feat": self.feat_tensor[:, :, split:],
            "raw": {k: v[:, split:] for k, v in self.raw_data_cache.items()},
            "target": self.target_ret[:, split:],
        }


# ─── 便捷入口 ────────────────────────────────────────────────
if __name__ == "__main__":
    loader = OKXDataLoader("BTC-USDT", "1H", 2000)
    loader.load_data()
    train, test = loader.train_test_split()
    print(f"Train: {train['feat'].shape}, Test: {test['feat'].shape}")
