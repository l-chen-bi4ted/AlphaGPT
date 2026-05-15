#!/usr/bin/env python3
"""
策略回顾（每整点 +3 分钟执行）
检查网格/信号状态，确认是否需要修正参数
"""
import sys, os, json, subprocess, time
import numpy as np

PROJ = os.path.expanduser("~/AlphaGPT-prod")
sys.path.insert(0, PROJ)

from signal_bot_offline import compute_adx


def fetch_candles(inst_id, limit=100):
    trade_pair = "BTC-USDT" if "BTC" in inst_id else "ETH-USDT"
    r = subprocess.run(
        ["okx","--demo","market","candles",trade_pair,"--bar","1H","--limit",str(limit)],
        capture_output=True, text=True, timeout=30
    )
    data = []
    for line in r.stdout.strip().split("\n"):
        parts = line.split()
        idx = next((i for i,p in enumerate(parts) if p.replace(".","",1).lstrip("-").isdigit()), None)
        if idx is None: continue
        pp = parts[idx:]
        if len(pp) >= 5:
            try: data.append({"high":float(pp[1]),"low":float(pp[2]),"close":float(pp[3]),"vol":float(pp[4])})
            except: pass
    return data[::-1]


def get_grid_info():
    r = subprocess.run(["okx","--demo","bot","grid","orders","--algoOrdType","grid"],
                       capture_output=True, text=True, timeout=15)
    lines = r.stdout.strip().split("\n")
    grids = {}
    for line in lines[4:]:  # skip header
        parts = line.split()
        if len(parts) >= 8:
            name = parts[1]
            grids[name] = {"state": parts[3], "pnl": float(parts[4]), "min": float(parts[7]), "max": float(parts[6])}
    return grids


def review():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 🔍 策略回顾")

    grids = get_grid_info()
    actions = []

    # --- BTC ---
    btc_c = fetch_candles("BTC", 80)
    if len(btc_c) > 30:
        h=np.array([x["high"] for x in btc_c[-60:]]); l=np.array([x["low"] for x in btc_c[-60:]])
        c=np.array([x["close"] for x in btc_c[-60:]])
        adx=compute_adx(h,l,c); price=c[-1]
        print(f"BTC: ${price:,.0f} | ADX {adx:.1f}")
        if "BTC-USDT" in grids:
            g = grids["BTC-USDT"]
            margin = price - g["min"]
            print(f"  Grid: PnL {g['pnl']*100:+.2f}% | Bottom margin: ${margin:,.0f}")
            if margin < 0:
                actions.append(f"[ACTION] BTC 跌破网格下界 $78,000, 建议下调至 $76,000")

    # --- ETH ---
    eth_c = fetch_candles("ETH", 80)
    if len(eth_c) > 30:
        h=np.array([x["high"] for x in eth_c[-60:]]); l=np.array([x["low"] for x in eth_c[-60:]])
        c=np.array([x["close"] for x in eth_c[-60:]])
        adx=compute_adx(h,l,c); price=c[-1]
        print(f"ETH: ${price:,.0f} | ADX {adx:.1f}")
        if "ETH-USDT" in grids:
            g = grids["ETH-USDT"]
            margin = price - g["min"]
            print(f"  Grid: PnL {g['pnl']*100:+.2f}% | Bottom margin: ${margin:,.0f}")
            if margin < 0:
                actions.append(f"[ACTION] ETH 已跌破网格下界 $2,200, 建议重建网格 $2,000-$2,400")

    # --- Execute adjustments ---
    for a in actions:
        print(f"\n⚠️  {a}")
        if "BTC" in a and "跌破" in a:
            print("  → 执行: 停止旧BTC网格, 新建 $76K-$85K 网格")
            # stop old
            subprocess.run(["okx","--demo","bot","grid","stop",
                "--algoId","3557226397120745472","--algoOrdType","grid","--instId","BTC-USDT"], timeout=15)
            time.sleep(1)
            # create new
            subprocess.run(["okx","--demo","bot","grid","create",
                "--instId","BTC-USDT","--algoOrdType","grid",
                "--maxPx","85000","--minPx","76000","--gridNum","20","--quoteSz","5000"], timeout=15)
            print("  ✅ BTC 网格重建完成")
        if "ETH" in a and "跌破" in a:
            print("  → 执行: 停止旧ETH网格, 新建 $2,000-$2,450 网格")
            subprocess.run(["okx","--demo","bot","grid","stop",
                "--algoId","3557371469908856832","--algoOrdType","grid","--instId","ETH-USDT",
                "--stopType","1"], timeout=15)
            time.sleep(1)
            subprocess.run(["okx","--demo","bot","grid","create",
                "--instId","ETH-USDT","--algoOrdType","grid",
                "--maxPx","2450","--minPx","2000","--gridNum","20","--quoteSz","3000"], timeout=15)
            print("  ✅ ETH 网格重建完成")

    if not actions:
        print("  ✅ 无需修正")

    print(f"[{now}] 🔍 回顾完成")

if __name__ == "__main__":
    review()
