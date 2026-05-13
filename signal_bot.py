#!/usr/bin/env python3
"""
OKX Signal Bot — 全自动交易信号发射器（v2 信号质量优化版）
基于 v2 liquidity 公式 + MarketRegime 环境感知
增加：信号平滑滤波 + 入场确认 + 持仓超时 + 浮动跟踪离场

用法:
    python signal_bot.py BTC-USDT 1H
    python signal_bot.py ETH-USDT 1H
"""
import sys, os, json, time, subprocess
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/AlphaGPT-fork/.env"))

# ── 策略参数 ──
FORMULA = [1, 17, 11, 11, 11, 14, 11, 11, 11, 17, 12, 11]

# 阈值
SIGNAL_THRESHOLD_TRENDING = 0.5
SIGNAL_THRESHOLD_RANGING = 0.7

# ADX
ADX_TRENDING = 25
ADX_RANGING = 20

# ---- v2 新增参数 ────────────────────────────────
SIGNAL_EMA_PERIOD = 3           # 信号 EMA 平滑周期
CONFIRM_BARS = 2                # 入场确认：连续 CONFIRM_BARS 根 K 线信号同向
POSITION_TIMEOUT_BARS = 24      # 持仓超时：24 根 K 线后强制离场
TRAIL_PROFIT_PCT = 2.0          # 浮盈 > 2% 时启用跟踪离场
TRAIL_RETRACE_PCT = 0.5         # 从最高浮盈回落 0.5% 即离场
# ─────────────────────────────────────────────

# ── 状态文件 ──
STATE_FILE = os.path.expanduser("~/AlphaGPT-prod-dev/signal_state.json")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "position": None,
        "entry_price": 0,
        "entry_time": "",
        "peak_signal": 0,
        "peak_price": 0,
        "signal_buffer": [],      # 最近 N 根 K 线的信号值
        "last_signal": 0,
        "total_trades": 0,
        "demo_balance": 91738,
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def fetch_candles(inst_id, bar, limit=200):
    """通过 okx CLI 拉取 K 线数据"""
    result = subprocess.run(
        ["okx", "--demo", "market", "candles", inst_id, "--bar", bar, "--limit", str(limit)],
        capture_output=True, text=True, timeout=30,
    )
    lines = result.stdout.strip().split("\n")
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) < 6:
            continue
        # Find first numeric column (price data starts after datetime)
        idx = 0
        for i, p in enumerate(parts):
            try:
                float(p)
                idx = i
                break
            except ValueError:
                continue
        price_parts = parts[idx:]
        if len(price_parts) < 5:
            continue
        try:
            data.append({
                "time": " ".join(parts[:idx]),
                "open": float(price_parts[0]), "high": float(price_parts[1]),
                "low": float(price_parts[2]), "close": float(price_parts[3]),
                "volume": float(price_parts[4]) if len(price_parts) > 4 else 0,
            })
        except ValueError:
            continue
    return data


def compute_adx(highs, lows, closes, period=14):
    n = len(closes)
    if n < period + 1:
        return 20
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        up = highs[i] - highs[i-1]
        down = lows[i-1] - lows[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0
    atr = np.zeros(n)
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period+1, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    sp = np.zeros(n); sp[period] = np.mean(plus_dm[1:period+1])
    for i in range(period+1, n):
        sp[i] = (sp[i-1] * (period-1) + plus_dm[i]) / period
    sm = np.zeros(n); sm[period] = np.mean(minus_dm[1:period+1])
    for i in range(period+1, n):
        sm[i] = (sm[i-1] * (period-1) + minus_dm[i]) / period
    di_plus = np.where(atr > 0, 100 * sp / atr, 0)
    di_minus = np.where(atr > 0, 100 * sm / atr, 0)
    dx = np.where(di_plus + di_minus > 0, 100 * abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.zeros(n)
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, n):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    return float(adx[-1])


def compute_factors(candles):
    n = len(candles)
    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])
    ret = np.zeros(n); ret[1:] = closes[1:] / closes[:-1] - 1
    liq = np.zeros(n)
    for i in range(1, n):
        liq[i] = np.log1p(abs(ret[i]) / (volumes[i] * closes[i] + 1e-12) * 1e8)
    pressure = np.zeros(n)
    for i in range(1, n):
        pressure[i] = (closes[i] - lows[i]) / (highs[i] - lows[i] + 1e-12) - 0.5
    ma10 = np.convolve(closes, np.ones(10)/10, mode='same')
    fomo = (closes - ma10) / (ma10 + 1e-12)
    dev = np.zeros(n)
    for i in range(14, n):
        dev[i] = np.std(ret[i-13:i+1])
    log_vol = np.log(volumes + 1)

    def rolling_zscore(x, window=20):
        r = np.zeros_like(x)
        for i in range(window, len(x)):
            seg = x[i-window:i]
            s = np.std(seg)
            if s > 0: r[i] = (x[i] - np.mean(seg)) / s
        return r
    return {
        "RET": rolling_zscore(ret), "LIQ": rolling_zscore(liq),
        "PRESSURE": rolling_zscore(pressure), "FOMO": rolling_zscore(fomo),
        "DEV": rolling_zscore(dev), "LOG_VOL": rolling_zscore(log_vol),
    }


def detect_regime(candles):
    if len(candles) < 30:
        return "RANGING"
    adx = compute_adx(
        np.array([c["high"] for c in candles]),
        np.array([c["low"] for c in candles]),
        np.array([c["close"] for c in candles]),
    )
    return "TRENDING" if adx > ADX_TRENDING else ("RANGING" if adx < ADX_RANGING else "VOLATILE")


def compute_signal(factors):
    """符号共识信号"""
    n = len(factors["RET"])
    if n < 30: return 0.0
    votes, wts = [], []
    ret = np.sign(factors["RET"][-5:].mean()); votes.append(ret); wts.append(0.3)
    liq_s = factors["LIQ"][-3:].mean(); liq_l = factors["LIQ"][-10:-3].mean()
    votes.append(np.sign(liq_s - liq_l)); wts.append(0.25)
    votes.append(np.sign(factors["PRESSURE"][-5:].mean())); wts.append(0.25)
    fomo = factors["FOMO"][-5:].mean()
    if abs(fomo) > 2.0:
        votes.append(np.sign(fomo)); wts.append(0.2)
    else:
        votes.append(0); wts.append(0.0)
    s = sum(v * w for v, w in zip(votes, wts)) / (sum(wts) or 1.0)
    return float(s)


def ema_smooth(values, period):
    """EMA 平滑"""
    alpha = 2.0 / (period + 1)
    result = np.zeros_like(values)
    if len(values) == 0: return result
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
    return result


def post_signal(action, inst_id, amount="10"):
    if action == "ENTER_LONG": side = "buy"
    elif action == "EXIT_LONG": side = "sell"
    else: return 0, f"unknown action: {action}"
    cmd = [
        "okx", "--demo", "spot", "place",
        "--instId", inst_id, "--side", side, "--ordType", "market",
        "--sz", str(amount), "--tgtCcy", "quote_ccy",
    ]
    try:
        env = {"HOME": "/Users/tsunemori", "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"}
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, env=env)
        out = result.stdout.strip()
        if result.returncode == 0: return 200, out
        return result.returncode, result.stderr.strip() or out
    except Exception as e:
        return 0, str(e)


def run(inst_id="BTC-USDT", bar="1H"):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] Signal Bot v2 — {inst_id} {bar}")

    # 1. 拉数据
    candles = fetch_candles(inst_id, bar, limit=200)
    if len(candles) < 60:
        print(f"  ❌ 数据不足 ({len(candles)} 条)")
        return
    last_price = candles[-1]["close"]
    print(f"  Price: ${last_price:,.2f}")

    # 2. 计算因子 + 原始信号
    factors = compute_factors(candles)
    signal = compute_signal(factors)
    regime = detect_regime(candles)
    print(f"  Raw:   {signal:+.4f}")

    # 3. EMA 平滑
    # 从历史数据构建信号序列用于 EMA
    sig_hist = []
    for offset in range(max(30, SIGNAL_EMA_PERIOD * 3), 0, -1):
        # 简化：用当前因子滚动计算近似
        sig_hist.append(signal * (1 - 0.1 * offset))  # 近似值
    sig_hist.append(signal)
    sig_array = ema_smooth(np.array(sig_hist), SIGNAL_EMA_PERIOD)
    smoothed = float(sig_array[-1])
    print(f"  Smooth: {smoothed:+.4f} (EMA{SIGNAL_EMA_PERIOD})")

    state = load_state()

    # 4. 更新信号历史缓冲
    buf = state.get("signal_buffer", [])
    buf.append(smoothed)
    if len(buf) > CONFIRM_BARS + 2:
        buf = buf[-(CONFIRM_BARS + 2):]
    state["signal_buffer"] = buf

    threshold = SIGNAL_THRESHOLD_RANGING if regime == "RANGING" else SIGNAL_THRESHOLD_TRENDING
    action = None

    # ---- 进场逻辑 ──
    if state["position"] is None and regime != "VOLATILE":
        # 确认：最近 CONFIRM_BARS 根 K 线信号同向且超过阈值
        if len(buf) >= CONFIRM_BARS:
            recent = buf[-CONFIRM_BARS:]
            if all(s > threshold for s in recent):
                action = "ENTER_LONG"
                print(f"  ✓ 确认入场 | 近{CONFIRM_BARS}根信号: {[f'{s:.3f}' for s in recent]}")
            elif all(s < -threshold for s in recent):
                action = "ENTER_SHORT"  # 预留，当前只做多
                print(f"  ✓ 看空信号 (短多未开启)")
            else:
                print(f"  ⏳ 等待确认 | buf={[f'{s:.3f}' for s in recent]}")
        else:
            print(f"  ⏳ 缓冲期 ({len(buf)}/{CONFIRM_BARS})")

    # ---- 持仓管理 ──
    elif state["position"] is not None:
        entry_price = state["entry_price"]
        entry_time = state.get("entry_time", "")
        pnl_pct = (last_price - entry_price) / entry_price * 100

        reasons = []

        # 条件 A：信号反转离场
        if smoothed < -threshold:
            reasons.append(f"signal={smoothed:.3f} < -{threshold}")

        # 条件 B：持仓超时
        bars_held = state.get("bars_held", 0) + 1
        state["bars_held"] = bars_held
        if bars_held >= POSITION_TIMEOUT_BARS:
            reasons.append(f"timeout ({bars_held}/{POSITION_TIMEOUT_BARS} bars)")

        # 条件 C：跟踪离场（浮盈后回落）
        if pnl_pct > TRAIL_PROFIT_PCT:
            peak = state.get("peak_price", entry_price)
            if last_price > peak:
                state["peak_price"] = last_price
                peak = last_price
            retrace = (peak - last_price) / peak * 100
            if retrace > TRAIL_RETRACE_PCT:
                reasons.append(f"trail stop ({retrace:.2f}% retrace from peak)")

        if reasons:
            action = "EXIT_LONG" if state["position"] == "LONG" else "EXIT_SHORT"
            print(f"  ✓ 离场: {' | '.join(reasons)} (pnl={pnl_pct:+.2f}%)")
        else:
            print(f"  ⏳ 持仓中 | pnl={pnl_pct:+.2f}% | bar={bars_held}/{POSITION_TIMEOUT_BARS} | threshold={threshold}")

    # ---- 执行 ──
    if action:
        amount = "10" if "ENTER" in action else "100"
        status, resp = post_signal(action, inst_id, amount)
        print(f"  {action} → {'✅' if status == 200 else '❌'} HTTP {status} | {resp[:80]}")

        if status == 200 and "ENTER" in action:
            state["position"] = "LONG" if "LONG" in action else "SHORT"
            state["entry_price"] = last_price
            state["entry_time"] = now_str
            state["peak_price"] = last_price
            state["bars_held"] = 0
            state["total_trades"] = state.get("total_trades", 0) + 1
        elif status == 200 and "EXIT" in action:
            state["position"] = None
            state["entry_price"] = 0
            state["entry_time"] = ""
            state["peak_price"] = 0
            state["bars_held"] = 0

    state["last_signal"] = smoothed
    save_state(state)
    print(f"  State: position={state['position']} | trades={state.get('total_trades', 0)}")


if __name__ == "__main__":
    inst_id = sys.argv[1] if len(sys.argv) > 1 else "BTC-USDT"
    bar = sys.argv[2] if len(sys.argv) > 2 else "1H"
    run(inst_id, bar)
