#!/usr/bin/env python3
"""
策略回顾（每整点 +3 分钟执行）
检查网格/信号/DCA状态，自动预警和修正参数。

规则：
  1. 距网格下界 < 2% 且 ADX > 25 → 预警（趋势击穿风险）
  2. ADX > 35 → 自动停止所有 DCA bot（马丁格尔不适用趋势市）
  3. ADX 从 < 20 突破到 > 25 → 趋势切换通知
  4. 价格跌破网格下界 → 自动重建网格（下移 $2K 缓冲）
"""
import sys, os, json, subprocess, time
import numpy as np

PROJ = os.path.expanduser("~/AlphaGPT-prod")
sys.path.insert(0, PROJ)
from signal_bot_offline import compute_adx

STATE_FILE = os.path.join(PROJ, ".review_state.json")

# ── 阈值 ──
GRID_MARGIN_WARN  = 0.02   # 距下界 < 2%
ADX_GRID_WARN     = 25     # ADX > 25 时触发网格预警
ADX_DCA_KILL      = 35     # ADX > 35 停所有 DCA
ADX_TREND_CROSS   = 25     # ADX 突破此线视为趋势切换
ADX_RANGE_EXIT    = 20     # ADX 从低于此值突破时记录


def load_state():
    """加载上次 ADX 快照"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except:
            pass
    return {}


def save_state(data):
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def fetch_candles(inst_id, limit=100):
    trade_pair = "BTC-USDT" if "BTC" in inst_id else "ETH-USDT"
    r = subprocess.run(
        ["okx", "--demo", "market", "candles", trade_pair, "--bar", "1H", "--limit", str(limit)],
        capture_output=True, text=True, timeout=30
    )
    data = []
    for line in r.stdout.strip().split("\n"):
        parts = line.split()
        idx = next((i for i, p in enumerate(parts)
                     if p.replace(".", "", 1).lstrip("-").isdigit()), None)
        if idx is None:
            continue
        pp = parts[idx:]
        if len(pp) >= 5:
            try:
                data.append({
                    "high": float(pp[1]),
                    "low": float(pp[2]),
                    "close": float(pp[3]),
                    "vol": float(pp[4]),
                })
            except:
                pass
    return data[::-1]


def get_grids():
    """获取所有运行中的网格，返回 {instId: {algoId, state, pnl, min, max}}"""
    r = subprocess.run(
        ["okx", "--demo", "bot", "grid", "orders", "--algoOrdType", "grid"],
        capture_output=True, text=True, timeout=15
    )
    grids = {}
    for line in r.stdout.strip().split("\n")[4:]:
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            inst_id = parts[1]
            grids[inst_id] = {
                "algoId": parts[0],
                "state": parts[3],
                "pnl": float(parts[4]),
                "max": float(parts[6]),
                "min": float(parts[7]),
            }
        except (ValueError, IndexError):
            continue
    return grids


def get_dca_bots():
    """获取所有运行中的 DCA bot"""
    r = subprocess.run(
        ["okx", "--demo", "bot", "dca", "orders"],
        capture_output=True, text=True, timeout=15
    )
    if "No DCA bots" in r.stdout:
        return []
    bots = []
    for line in r.stdout.strip().split("\n")[4:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            bots.append({"algoId": parts[0], "instId": parts[1], "state": parts[3]})
        except (ValueError, IndexError):
            continue
    return bots


def compute_market(inst_id, candles):
    """计算 ADX 和当前价格"""
    if len(candles) < 60:
        return None, None
    h = np.array([x["high"] for x in candles[-60:]])
    l = np.array([x["low"] for x in candles[-60:]])
    c = np.array([x["close"] for x in candles[-60:]])
    adx = compute_adx(h, l, c)
    return float(adx), float(c[-1])


def stop_and_rebuild_grid(inst_id, old_grid, new_min, new_max):
    """停止旧网格并重建"""
    print(f"  ⚙️  停止旧网格 {old_grid['algoId']} …")
    subprocess.run([
        "okx", "--demo", "bot", "grid", "stop",
        "--algoId", old_grid["algoId"],
        "--algoOrdType", "grid",
        "--instId", inst_id,
    ], capture_output=True, timeout=15)
    time.sleep(1)

    quote_sz = "5000" if "BTC" in inst_id else "3000"
    print(f"  ⚙️  新建网格: ${new_min:,}-${new_max:,}")
    subprocess.run([
        "okx", "--demo", "bot", "grid", "create",
        "--instId", inst_id,
        "--algoOrdType", "grid",
        "--maxPx", str(new_max),
        "--minPx", str(new_min),
        "--gridNum", "20",
        "--quoteSz", quote_sz,
    ], capture_output=True, timeout=15)
    print(f"  ✅ {inst_id} 网格重建完成")


def stop_all_dca():
    """停止所有 DCA bot"""
    bots = get_dca_bots()
    if not bots:
        return 0
    stopped = 0
    for b in bots:
        print(f"  ⛔ 停止 DCA {b['instId']} ({b['algoId']}) …")
        r = subprocess.run([
            "okx", "--demo", "bot", "dca", "stop",
            "--algoId", b["algoId"],
        ], capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            stopped += 1
        else:
            print(f"    ⚠️  停止失败: {r.stderr.strip()[:100]}")
    return stopped


def review():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 🔍 策略回顾 — 强化版")

    state = load_state()
    grids = get_grids()
    actions = []
    new_state = {}

    # ── 遍历 BTC / ETH ──
    for label, inst_id, buffer_amount in [
        ("BTC", "BTC-USDT", 2000),
        ("ETH", "ETH-USDT", 200),
    ]:
        candles = fetch_candles(label, 80)
        adx, price = compute_market(label, candles)

        if adx is None:
            continue

        prev_adx = state.get(f"{label}_ADX", adx)
        new_state[f"{label}_ADX"] = adx

        regime = "TREND" if adx > ADX_GRID_WARN else "RANGE"
        direction = "📈" if len(candles) > 30 and candles[-1]["close"] > candles[-30]["close"] else "📉"
        print(f"{label}: ${price:,.0f} | ADX {adx:.1f} ({regime}) {direction}")

        # ── 规则 3: 趋势切换通知 ──
        if prev_adx < ADX_RANGE_EXIT and adx > ADX_TREND_CROSS:
            print(f"  🔔 趋势切换: ADX {prev_adx:.1f} → {adx:.1f} (震荡→趋势)")
            actions.append({
                "type": "alert",
                "msg": f"{label} 趋势切换: ADX {prev_adx:.1f}→{adx:.1f}，震荡→趋势",
            })

        # ── 规则 1: 网格下界预警 ──
        if inst_id in grids:
            g = grids[inst_id]
            margin_frac = (price - g["min"]) / g["min"]
            margin_abs = price - g["min"]
            print(f"  Grid: PnL {g['pnl']*100:+.2f}% | 距下界 {margin_frac*100:+.1f}% (${margin_abs:+,.0f})")

            if margin_frac < GRID_MARGIN_WARN and adx > ADX_GRID_WARN:
                print(f"  ⚠️  距下界仅 {margin_frac*100:.1f}% 且 ADX={adx:.1f}>25，趋势击穿风险！")
                actions.append({
                    "type": "warn",
                    "msg": f"{label} 距网格下界 {margin_frac*100:.1f}%，ADX={adx:.1f}，建议下调下界",
                })

            # ── 规则 4: 跌破下界自动重建 ──
            if margin_abs < 0:
                new_min = int(g["min"] - buffer_amount)
                new_max = int(g["max"])
                print(f"  🚨 {label} 已跌破网格下界 ${g['min']:,.0f}！")
                actions.append({
                    "type": "rebuild",
                    "inst_id": inst_id,
                    "old_grid": g,
                    "new_min": new_min,
                    "new_max": new_max,
                })

    # ── 规则 2: ADX > 35 停所有 DCA ──
    btc_adx = state.get("BTC_ADX", 0)
    eth_adx = state.get("ETH_ADX", 0)
    max_adx = max(btc_adx, eth_adx)  # 用旧值比较，因为 DCA 规则用新值判断

    # 用当前新 ADX 值判断
    btc_adx_now = new_state.get("BTC_ADX", 0)
    eth_adx_now = new_state.get("ETH_ADX", 0)
    max_adx_now = max(btc_adx_now, eth_adx_now)

    if max_adx_now > ADX_DCA_KILL:
        dca_bots = get_dca_bots()
        if dca_bots:
            print(f"\n🚨 ADX={max_adx_now:.1f}>{ADX_DCA_KILL}，触发 DCA 清退！")
            stopped = stop_all_dca()
            print(f"⛔ 已停止 {stopped} 个 DCA bot")
        elif max_adx > ADX_DCA_KILL:  # 之前就高于35，不重复报
            pass  # 已经在高 ADX，不重复
        else:
            print(f"\n⚠️  ADX={max_adx_now:.1f}>{ADX_DCA_KILL} — 无 DCA bot 运行，跳过")

    # ── 执行修正动作 ──
    for a in actions:
        if a["type"] == "alert":
            print(f"\n🔔 {a['msg']}")
        elif a["type"] == "warn":
            print(f"\n⚠️  {a['msg']}")
        elif a["type"] == "rebuild":
            inst_id = a["inst_id"]
            print(f"\n🚨 {inst_id} 跌破下界，自动重建…")
            stop_and_rebuild_grid(inst_id, a["old_grid"], a["new_min"], a["new_max"])

    if not actions:
        print("  ✅ 无异常")

    save_state(new_state)
    print(f"[{now}] 🔍 回顾完成")


if __name__ == "__main__":
    review()
