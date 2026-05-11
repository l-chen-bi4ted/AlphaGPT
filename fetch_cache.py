"""
拉取 OKX K 线数据到本地 CSV 缓存。
history-candles 拉历史（>2天前），candles 拉近2天，自动切换。
用法：python fetch_cache.py
输出：data_cache/*.csv
"""
import time, os, json, hashlib
import requests
import pandas as pd

OKX = "https://www.okx.com"
HDR = {"User-Agent": "Mozilla/5.0 Chrome/120.0"}
CACHE = os.path.join(os.path.dirname(__file__), "data_cache")
SYMBOLS = [
    ("BTC-USDT", "1H", 5000),
    ("ETH-USDT", "1H", 5000),
    ("SOL-USDT", "1H", 5000),
]


def _get(path, params, timeout=20):
    for n in range(4):
        try:
            r = requests.get(f"{OKX}{path}", params=params, timeout=timeout, headers=HDR)
            if r.status_code == 429:
                time.sleep(2 ** n); continue
            r.raise_for_status()
            b = r.json()
            if b.get("code") != "0":
                if n < 3: time.sleep(1); continue
                return []
            return [{
                "ts": int(d[0]), "open": float(d[1]), "high": float(d[2]),
                "low": float(d[3]), "close": float(d[4]), "vol": float(d[5]),
                "vol_ccy": float(d[6]) if len(d) > 6 and d[6] else float(d[5]) * float(d[4]),
            } for d in b.get("data", [])]
        except Exception as e:
            if n >= 3: print(f"\n  err: {e}"); return []
            time.sleep(1)
    return []


def fetch_all(inst_id, bar, limit):
    rows, after = [], ""
    # Phase 1: history (>2 days ago)
    while len(rows) < limit:
        page = _get("/api/v5/market/history-candles",
                     {"instId": inst_id, "bar": bar, "limit": 100, "after": after} if after else
                     {"instId": inst_id, "bar": bar, "limit": 100})
        if not page: break
        rows.extend(page)
        if len(page) < 100: break
        after = str(page[-1]["ts"]); time.sleep(0.12)

    # Phase 2: recent (<2 days) — pull once
    recent = _get("/api/v5/market/candles",
                   {"instId": inst_id, "bar": bar, "limit": 300})
    if recent: rows.extend(recent)

    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts").tail(limit)


def _write_meta(fp: str, df: pd.DataFrame, inst_id: str, bar: str):
    content = df.to_csv(index=False)
    meta = {
        "inst_id": inst_id,
        "bar": bar,
        "rows": len(df),
        "ts_min": int(df["ts"].min()) if len(df) else None,
        "ts_max": int(df["ts"].max()) if len(df) else None,
        "sha256": hashlib.sha256(content.encode()).hexdigest()[:16],
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_fp = fp.replace(".csv", ".meta.json")
    with open(meta_fp, "w") as f:
        json.dump(meta, f, indent=2)


os.makedirs(CACHE, exist_ok=True)
for inst_id, bar, limit in SYMBOLS:
    fn = f"{inst_id.replace('-', '')}_{bar}.csv"
    fp = os.path.join(CACHE, fn)
    print(f"[fetch] {inst_id} {bar} ", end="", flush=True)
    df = fetch_all(inst_id, bar, limit)
    df.to_csv(fp, index=False)
    _write_meta(fp, df, inst_id, bar)
    print(f"→ {len(df)} rows")

print(f"\nDone. {CACHE}:")
for f in sorted(os.listdir(CACHE)):
    print(f"  {f}  ({os.path.getsize(os.path.join(CACHE, f)):,} B)")
