#!/usr/bin/env python3
"""
Offline Signal Bot — 读取本地缓存 CSV 计算信号，无需网络。
用于网络受限环境的信号计算（如中国大陆服务器）。

用法: python3 signal_bot_offline.py BTC-USDT 1H
"""
import sys, os, json, csv
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(__file__), "data_cache", "BTCUSDT_1H.csv")
STATE_FILE = os.path.join(os.path.dirname(__file__), "signal_state.json")

# ── 策略参数 ──
SIGNAL_THRESHOLD_TRENDING = 0.5
SIGNAL_THRESHOLD_RANGING = 0.7
ADX_TRENDING = 25
ADX_RANGING = 20
SIGNAL_EMA_PERIOD = 3
CONFIRM_BARS = 2


def load_candles(csv_path):
    """从本地 CSV 读取 K 线数据"""
    if not os.path.exists(csv_path):
        print(f"❌ 缓存文件未找到: {csv_path}")
        sys.exit(1)
    data = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data.append({
                    "time": row.get("ts", ""),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("vol", row.get("volume", 0))),
                })
            except (ValueError, KeyError):
                continue
    return data


def compute_adx(highs, lows, closes, period=14):
    n = len(closes)
    if n < period + 1:
        return 20
    tr = np.zeros(n)
    pdm = np.zeros(n); ndm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        up = highs[i] - highs[i-1]; down = lows[i-1] - lows[i]
        pdm[i] = up if up > down and up > 0 else 0
        ndm[i] = down if down > up and down > 0 else 0
    atr = np.zeros(n); atr[period] = np.mean(tr[1:period+1])
    sp = np.zeros(n); sp[period] = np.mean(pdm[1:period+1])
    sm = np.zeros(n); sm[period] = np.mean(ndm[1:period+1])
    for i in range(period+1, n):
        atr[i] = (atr[i-1]*(period-1)+tr[i])/period
        sp[i] = (sp[i-1]*(period-1)+pdm[i])/period
        sm[i] = (sm[i-1]*(period-1)+ndm[i])/period
    di_plus = np.where(atr > 0, 100*sp/atr, 0)
    di_minus = np.where(atr > 0, 100*sm/atr, 0)
    dx = np.where(di_plus+di_minus > 0, 100*abs(di_plus-di_minus)/(di_plus+di_minus), 0)
    adx = np.zeros(n); adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, n):
        adx[i] = (adx[i-1]*(period-1)+dx[i])/period
    return float(adx[-1])


def compute_factors(candles):
    n = len(candles)
    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])
    ret = np.zeros(n); ret[1:] = closes[1:]/closes[:-1]-1
    liq = np.zeros(n)
    for i in range(1, n):
        liq[i] = np.log1p(abs(ret[i])/(volumes[i]*closes[i]+1e-12)*1e8)
    pressure = np.zeros(n)
    for i in range(1, n):
        pressure[i] = (closes[i]-lows[i])/(highs[i]-lows[i]+1e-12)-0.5
    ma10 = np.convolve(closes, np.ones(10)/10, mode='same')
    fomo = (closes-ma10)/(ma10+1e-12)
    def rolling_zscore(x, window=20):
        r = np.zeros_like(x)
        for i in range(window, len(x)):
            seg = x[i-window:i]; s = np.std(seg)
            if s > 0: r[i] = (x[i]-np.mean(seg))/s
        return r
    return {"RET": rolling_zscore(ret), "LIQ": rolling_zscore(liq),
            "PRESSURE": rolling_zscore(pressure), "FOMO": rolling_zscore(fomo)}


def compute_signal(factors):
    if len(factors["RET"]) < 30: return 0.0
    votes, wts = [], []
    ret = np.sign(factors["RET"][-5:].mean()); votes.append(ret); wts.append(0.35)
    liq_s = factors["LIQ"][-3:].mean(); liq_l = factors["LIQ"][-10:-3].mean()
    votes.append(np.sign(liq_s-liq_l)); wts.append(0.25)
    votes.append(np.sign(factors["PRESSURE"][-5:].mean())); wts.append(0.25)
    fomo = factors["FOMO"][-5:].mean()
    if abs(fomo) > 2.0: votes.append(np.sign(fomo)); wts.append(0.15)
    else: votes.append(0); wts.append(0.0)
    return float(sum(v*w for v,w in zip(votes,wts))/(sum(wts) or 1.0))


def ema(values, period):
    a = 2.0/(period+1); r = np.zeros_like(values)
    if len(values) == 0: return r
    r[0] = values[0]
    for i in range(1, len(values)): r[i] = a*values[i]+(1-a)*r[i-1]
    return r


def detect_regime(candles):
    if len(candles) < 30: return "RANGING"
    adx = compute_adx(np.array([c["high"] for c in candles]),
                      np.array([c["low"] for c in candles]),
                      np.array([c["close"] for c in candles]))
    return "TRENDING" if adx > ADX_TRENDING else ("RANGING" if adx < ADX_RANGING else "VOLATILE")


def main(csv_path=CSV_PATH):
    candles = load_candles(csv_path)
    ts_range = f"{candles[0]['time']} → {candles[-1]['time']}" if len(candles) > 0 else "empty"
    print(f"📊 K线: {len(candles)} 条 | {ts_range}")
    print(f"💰 最新价: ${candles[-1]['close']:,.2f}" if candles else "")

    factors = compute_factors(candles)
    signal = compute_signal(factors)
    regime = detect_regime(candles)

    sig_hist = [signal*(1-0.1*o) for o in range(max(30, SIGNAL_EMA_PERIOD*3), 0, -1)] + [signal]
    smoothed = float(ema(np.array(sig_hist), SIGNAL_EMA_PERIOD)[-1])

    threshold = SIGNAL_THRESHOLD_RANGING if regime == "RANGING" else SIGNAL_THRESHOLD_TRENDING

    print(f"📈 原始信号: {signal:+.4f}")
    print(f"📉 平滑信号: {smoothed:+.4f} (EMA{SIGNAL_EMA_PERIOD})")
    print(f"🌡️  市场状态: {regime} (阈值={threshold})")

    if regime == "VOLATILE":
        print(f"⏸️  VOLATILE — 暂停交易")
    elif abs(smoothed) > threshold:
        action = "ENTER_LONG" if smoothed > 0 else "ENTER_SHORT"
        print(f"🚀 建议操作: {action} (|signal|={abs(smoothed):.4f} > {threshold})")
    else:
        print(f"⏸️  无操作 (|signal|={abs(smoothed):.4f} < {threshold})")

    print("\n━━━ 因子明细 ━━━")
    for name in ["RET","LIQ","PRESSURE","FOMO"]:
        v = factors[name][-5:].mean()
        print(f"  {name}: {v:+.4f}")


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    main(csv_file)
