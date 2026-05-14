#!/usr/bin/env python3
"""
每小时执行器：三池信号 + DCA 机会检测 + iMessage 通知
"""
import sys, os, json, subprocess, csv, time
import numpy as np

PROJ = os.path.expanduser("~/AlphaGPT-prod")
sys.path.insert(0, PROJ)


def run_script(name, *args):
    venv_python = os.path.join(PROJ, "venv", "bin", "python3")
    script = os.path.join(PROJ, name)
    cmd = [venv_python, script] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout, result.stderr


def get_adx_and_price():
    from signal_bot_offline import compute_adx
    csv_path = os.path.join(PROJ, "data_cache", "BTCUSDT_1H.csv")
    if not os.path.exists(csv_path):
        return None, None
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({"high": float(r["high"]), "low": float(r["low"]), "close": float(r["close"])})
    fetch = subprocess.run(
        ["okx", "--profile", "hermes-demo", "market", "candles", "BTC-USDT", "--bar", "1H", "--limit", "5"],
        capture_output=True, text=True, timeout=30
    )
    for line in fetch.stdout.strip().split("\n"):
        parts = line.split()
        idx = 0
        for i, p in enumerate(parts):
            try:
                float(p)
                idx = i
                break
            except ValueError:
                continue
        pp = parts[idx:]
        if len(pp) >= 5:
            try:
                rows.append({"high": float(pp[1]), "low": float(pp[2]), "close": float(pp[3])})
            except ValueError:
                pass
    if len(rows) < 30:
        return None, None
    c = rows[-60:]
    adx = compute_adx(
        np.array([x["high"] for x in c]),
        np.array([x["low"] for x in c]),
        np.array([x["close"] for x in c]),
    )
    return adx, c[-1]["close"]


def send_imessage(text):
    pass


def deploy_dca():
    cmd = [
        "okx", "--profile", "hermes-demo", "bot", "dca", "create",
        "--algoOrdType", "spot_dca",
        "--instId", "BTC-USDT",
        "--direction", "long",
        "--initOrdAmt", "200",
        "--maxSafetyOrds", "3",
        "--safetyOrdAmt", "200",
        "--volMult", "1.5",
        "--pxSteps", "0.02",
        "--pxStepsMult", "1.0",
        "--tpPct", "2.0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    return result.stdout.strip() or result.stderr.strip()


def main():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] start")

    print("-- spot --")
    out, _ = run_script("signal_bot.py", "BTC-USDT", "1H")
    for line in out.split("\n"):
        if "Price" in line or "Signal" in line or "Smooth" in line or "State" in line or "操作" in line or "等待" in line:
            print(f"  {line.strip()}")

    print("-- swap --")
    out2, _ = run_script("signal_bot_swap.py")
    for line in out2.split("\n"):
        if "Price" in line or "Signal" in line or "position" in line or "action" in line or "No action" in line:
            print(f"  {line.strip()}")

    print("-- DCA --")
    adx, price = get_adx_and_price()
    if adx is None:
        print("  skip (no data)")
    elif adx < 20:
        print(f"  ADX {adx:.1f} < 20 (RANGING) @ ${price:,.0f}")
        result = deploy_dca()
        if "created" in result.lower() or "OK" in result or "356" in result:
            msg = f"DCA bot live | BTC ${price:,.0f} | ADX {adx:.1f} | 200U x 3 @ 2%"
            print(f"  OK: {msg}")
            send_imessage(msg)
        else:
            print(f"  FAIL: {result[:80]}")
    else:
        print(f"  ADX {adx:.1f} >= 20, skip DCA @ ${price:,.0f}")

    print(f"[{now}] done")


if __name__ == "__main__":
    main()
