"""
OKX Market Data Connector — 替代 PostgreSQL/CryptoDataLoader。

从 OKX REST API 拉取 K 线数据，输出兼容 FeatureEngineer 的 raw_data_cache。
单标的模式：[1, T] 张量；多标的需要将 shape 从 [N, T] 传递给 FeatureEngineer。

OKX 公开行情 API 无需鉴权，限频 20 req / 2s。
"""

import time
import json
import os
import hashlib
from typing import Optional
from pathlib import Path

import numpy as np
import torch
import requests
import pandas as pd

from model_core.config import ModelConfig, default_config

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
        except requests.RequestException as e:
            print(f"  Fetch error (attempt {attempt+1}): {e}")
            time.sleep(1)
        except Exception as e:
            print(f"  Unexpected error (attempt {attempt+1}): {e}")
            time.sleep(1)
    return []


def fetch_all_candles(
    inst_id: str,
    bar: str = "1H",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    拉取 instId 的全部历史 K 线，返回 DataFrame (ts, o, h, l, c, vol, volCcy)。
    
    内部自动分页，先拉历史，再补近 2 天数据。
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
        # 用最后一条的 ts 作为下一页 after，但去重避免无限循环
        last_ts = page[-1]["ts"]
        if str(last_ts) == after:
            break
        after = str(last_ts)
        time.sleep(0.1)

    # 再拉近 2 天数据（可能重叠，后续去重）
    recent = _fetch_candles_page(inst_id, bar, limit=300, use_history=False)
    if recent:
        all_rows.extend(recent)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    
    # 连续性检查
    if len(df) > 1:
        bar_ms = {
            "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
            "30m": 1_800_000, "1H": 3_600_000, "2H": 7_200_000,
            "4H": 14_400_000, "6H": 21_600_000, "12H": 43_200_000, "1D": 86_400_000,
        }.get(bar, 3_600_000)
        gaps = df["ts"].diff().dropna()
        expected = bar_ms
        gap_count = (gaps > expected * 1.5).sum()
        if gap_count > 0:
            print(f"  ⚠️  Data gap detected: {gap_count} bars missing (expected ~{expected}ms interval)")
    
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
        config: Optional[ModelConfig] = None,
    ):
        self.inst_id = inst_id
        self.bar = bar
        self.limit = limit
        self.train_ratio = train_ratio
        self.config = config or default_config
        
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "data_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.feat_tensor: Optional[torch.Tensor] = None
        self.raw_data_cache: Optional[dict] = None
        self.target_ret: Optional[torch.Tensor] = None
        self.full_df: Optional[pd.DataFrame] = None
        self.meta: Optional[dict] = None  # 数据元信息（版本、校验、下载时间）

    def _cache_path(self) -> Path:
        """本地缓存文件路径：data_cache/BTCUSDT_1H.csv"""
        fname = f"{self.inst_id.replace('-', '')}_{self.bar}.csv"
        return self.cache_dir / fname

    def _meta_path(self) -> Path:
        """元数据路径：data_cache/BTCUSDT_1H.meta.json"""
        return self.cache_dir / f"{self.inst_id.replace('-', '')}_{self.bar}.meta.json"

    def _compute_meta(self, df: pd.DataFrame) -> dict:
        """计算数据元信息用于版本校验。"""
        content = df.to_csv(index=False)
        return {
            "inst_id": self.inst_id,
            "bar": self.bar,
            "rows": len(df),
            "ts_min": int(df["ts"].min()) if len(df) else None,
            "ts_max": int(df["ts"].max()) if len(df) else None,
            "sha256": hashlib.sha256(content.encode()).hexdigest()[:16],
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def load_data(self, force_refresh: bool = False):
        """优先读本地 CSV 缓存；无缓存则联网拉取。"""
        cache_path = self._cache_path()
        meta_path = self._meta_path()

        if not force_refresh and cache_path.exists():
            print(f"Loading from cache: {cache_path}")
            df = pd.read_csv(cache_path)
            
            # 校验元数据
            if meta_path.exists():
                with open(meta_path) as f:
                    stored_meta = json.load(f)
                current_meta = self._compute_meta(df)
                if stored_meta.get("sha256") != current_meta["sha256"]:
                    print("  ⚠️  Cache metadata mismatch, forcing refresh...")
                    force_refresh = True
            else:
                print("  ⚠️  No metadata found, will refresh...")
                force_refresh = True

        if force_refresh or not cache_path.exists():
            print(f"Fetching {self.inst_id} {self.bar} candles from OKX...")
            df = fetch_all_candles(self.inst_id, self.bar, self.limit)
            if df.empty:
                raise RuntimeError(f"No data for {self.inst_id}")
            df.to_csv(cache_path, index=False)
            meta = self._compute_meta(df)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"  Cached {len(df)} rows to {cache_path}")
        else:
            df = pd.read_csv(cache_path)

        print(f"  Got {len(df)} candles: {df['ts'].min()} → {df['ts'].max()}")
        self.full_df = df
        self.meta = self._compute_meta(df)

        # 转为张量 [1, T]
        device = self.config.device
        t = lambda col: torch.tensor(df[col].values, dtype=torch.float32, device=device).unsqueeze(0)

        vol_usd = torch.tensor(
            df["vol_ccy"].values, dtype=torch.float32, device=device
        ).unsqueeze(0)

        self.raw_data_cache = {
            "open":      t("open"),
            "high":      t("high"),
            "low":       t("low"),
            "close":     t("close"),
            "volume":    t("vol"),
            "liquidity": vol_usd,
            "fdv":       torch.full_like(vol_usd, 1e12),
        }

        # 特征张量 [F, 1, T]
        from model_core.factors import FeatureEngineer
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)

        # 目标：下期收益率（t 时刻因子预测 t→t+1 收益，对齐实盘推理）
        # 注意：target_ret[t] = log(close[t+1] / close[t])
        # 最后一个时间点的目标设为 0（无法预测未来）
        close = self.raw_data_cache["close"]  # [1, T]
        t1 = torch.cat([close[:, 1:], torch.zeros_like(close[:, :1])], dim=1)
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
