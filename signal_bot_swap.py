#!/usr/bin/env python3
"""
OKX Signal Bot — 合约版（做多 + 做空）
基于多因子共识 + ADX 环境感知
用于模拟盘合约练手，同时吃多空双向信号

用法:
    python3 signal_bot_swap.py BTC-USDT-SWAP 1H
"""
import sys, os, json, time, subprocess, csv
import numpy as np

load_dotenv = lambda p: None
if os.path.exists(os.path.expanduser("~/.hermes/.env")):
    from dotenv import load_dotenv; load_dotenv()

INST_ID = "BTC-USDT-SWAP"
BAR = "1H"
STATE_FILE = os.path.join(os.path.dirname(__file__), "signal_swap_state.json")
CSV_CACHE = os.path.join(os.path.dirname(__file__), "data_cache", "BTCUSDT_1H.csv")

# ── 策略参数 ──
SIGNAL_THRESHOLD_TRENDING = 0.5
SIGNAL_THRESHOLD_RANGING = 0.7
ADX_TRENDING = 25
ADX_RANGING = 20
SIGNAL_EMA_PERIOD = 3
CONFIRM_BARS = 2
POSITION_TIMEOUT_BARS = 24
TRAIL_PROFIT_PCT = 2.0
TRAIL_RETRACE_PCT = 0.5
CONTRACT_AMOUNT = 1      # 1 contract = 0.01 BTC
LEVERAGE = 3


def fetch_candles(limit=300):
    """通过 okx CLI 抓取 K 线"""
    r = subprocess.run(["okx", "--profile", "hermes-demo", "market", "candles",
                        "BTC-USDT", "--bar", BAR, "--limit", str(limit)],
                       capture_output=True, text=True, timeout=30)
    lines = r.stdout.strip().split("\n")
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) < 6: continue
        idx = next((i for i, p in enumerate(parts) if p.replace(".","",1).lstrip("-").isdigit()), -1)
        if idx < 0: continue
        pp = parts[idx:]
        if len(pp) < 5: continue
        try:
            data.append({"close": float(pp[3]), "high": float(pp[1]), "low": float(pp[2]),
                         "vol": float(pp[4]), "open": float(pp[0])})
        except: continue
    return data[::-1]  # reverse to oldest-first


def compute_adx(highs, lows, closes, period=14):
    n = len(closes)
    if n < period + 1: return 20
    tr, pdm, ndm = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        up = highs[i]-highs[i-1]; down = lows[i-1]-lows[i]
        pdm[i] = up if up>down and up>0 else 0
        ndm[i] = down if down>up and down>0 else 0
    atr = np.zeros(n); atr[period] = np.mean(tr[1:period+1])
    sp = np.zeros(n); sp[period] = np.mean(pdm[1:period+1])
    sm = np.zeros(n); sm[period] = np.mean(ndm[1:period+1])
    for i in range(period+1, n):
        atr[i]=(atr[i-1]*(period-1)+tr[i])/period; sp[i]=(sp[i-1]*(period-1)+pdm[i])/period; sm[i]=(sm[i-1]*(period-1)+ndm[i])/period
    di_plus = np.where(atr>0,100*sp/atr,0); di_minus = np.where(atr>0,100*sm/atr,0)
    dx = np.where(di_plus+di_minus>0,100*abs(di_plus-di_minus)/(di_plus+di_minus),0)
    adx = np.zeros(n); adx[2*period-1]=np.mean(dx[period:2*period])
    for i in range(2*period,n): adx[i]=(adx[i-1]*(period-1)+dx[i])/period
    return float(adx[-1])


def compute_factors(candles):
    n = len(candles)
    c = np.array([x["close"] for x in candles])
    h = np.array([x["high"] for x in candles])
    l = np.array([x["low"] for x in candles])
    v = np.array([x["vol"] for x in candles])
    ret = np.zeros(n); ret[1:] = c[1:]/c[:-1]-1
    liq = np.zeros(n)
    for i in range(1, n): liq[i] = np.log1p(abs(ret[i])/(v[i]*c[i]+1e-12)*1e8)
    pressure = np.zeros(n)
    for i in range(1, n): pressure[i] = (c[i]-l[i])/(h[i]-l[i]+1e-12)-0.5
    ma10 = np.convolve(c,np.ones(10)/10,mode='same')
    fomo = (c-ma10)/(ma10+1e-12)
    def zs(x,w=20):
        r=np.zeros_like(x)
        for i in range(w,len(x)):
            s=np.std(x[i-w:i])
            if s>0: r[i]=(x[i]-np.mean(x[i-w:i]))/s
        return r
    return {"RET":zs(ret),"LIQ":zs(liq),"PRESSURE":zs(pressure),"FOMO":zs(fomo)}


def compute_signal(factors):
    if len(factors["RET"])<30: return 0.0
    votes,wts=[],[]
    ret=np.sign(factors["RET"][-5:].mean()); votes.append(ret); wts.append(0.35)
    liq_s=factors["LIQ"][-3:].mean(); liq_l=factors["LIQ"][-10:-3].mean()
    votes.append(np.sign(liq_s-liq_l)); wts.append(0.25)
    votes.append(np.sign(factors["PRESSURE"][-5:].mean())); wts.append(0.25)
    fomo=factors["FOMO"][-5:].mean()
    if abs(fomo)>2.0: votes.append(np.sign(fomo)); wts.append(0.15)
    else: votes.append(0); wts.append(0.0)
    return float(sum(v*w for v,w in zip(votes,wts))/(sum(wts)or 1.0))


def ema(values, period=3):
    a=2.0/(period+1); r=np.zeros_like(values)
    if len(values)==0: return r
    r[0]=values[0]
    for i in range(1,len(values)): r[i]=a*values[i]+(1-a)*r[i-1]
    return r


def detect_regime(candles):
    if len(candles)<30: return "RANGING"
    adx=compute_adx(np.array([c["high"] for c in candles]),np.array([c["low"] for c in candles]),np.array([c["close"] for c in candles]))
    return "TRENDING" if adx>ADX_TRENDING else ("RANGING" if adx<ADX_RANGING else "VOLATILE")


def place_swap(side, amount=CONTRACT_AMOUNT):
    """下单"""
    cmd=["okx","--profile","hermes-demo","swap","place","--instId",INST_ID,"--side",side,
         "--ordType","market","--sz",str(amount),"--tdMode","cross"]
    if side in ("sell","buy") and os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f: s=json.load(f)
            if s.get("position") is not None: cmd.append("--reduceOnly")
        except: pass
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
        if r.returncode==0: return 200, r.stdout.strip()
        return r.returncode, r.stderr.strip() or r.stdout.strip()
    except Exception as e: return 0, str(e)


def run():
    now=time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Swap Signal Bot — {INST_ID} {BAR}")

    candles = fetch_candles()
    if len(candles)<60: print(f"❌ 数据不足 ({len(candles)})"); return
    price = candles[-1]["close"]
    print(f"Price: ${price:,.2f}")

    factors=compute_factors(candles); sig=compute_signal(factors)
    regime=detect_regime(candles)
    hist=[sig*(1-0.1*o) for o in range(30,0,-1)]+[sig]
    smooth=float(ema(np.array(hist),SIGNAL_EMA_PERIOD)[-1])
    threshold=SIGNAL_THRESHOLD_RANGING if regime=="RANGING" else SIGNAL_THRESHOLD_TRENDING
    print(f"Signal: {sig:+.4f} → smooth {smooth:+.4f} | {regime} (thresh={threshold})")

    state={}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: state=json.load(f)
    state.setdefault("position",None); state.setdefault("entry_price",0)
    state.setdefault("bars_held",0); state.setdefault("peak_price",0)

    action=None
    if state["position"]is None and regime!="VOLATILE":
        if smooth>threshold: action="LONG_ENTRY"
        elif smooth<-threshold: action="SHORT_ENTRY"
    elif state["position"]is not None:
        pnl=(price-state["entry_price"])/state["entry_price"]*100
        reasons=[]
        if state["position"]=="LONG":
            if smooth<-threshold: reasons.append(f"signal={smooth:.3f}<-{threshold}")
            if smooth>threshold: pass
        elif state["position"]=="SHORT":
            if smooth>threshold: reasons.append(f"signal={smooth:.3f}>{threshold}")
            if smooth<-threshold: pass
        state["bars_held"]+=1
        if state["bars_held"]>=POSITION_TIMEOUT_BARS: reasons.append(f"timeout({state['bars_held']})")
        if pnl>TRAIL_PROFIT_PCT:
            peak=state.get("peak_price",state["entry_price"])
            if price>peak: state["peak_price"]=peak=price
            if (peak-price)/peak*100>TRAIL_RETRACE_PCT: reasons.append(f"trail({(peak-price)/peak*100:.2f}%)")
        if reasons:
            action="LONG_EXIT" if state["position"]=="LONG" else "SHORT_EXIT"
            print(f"Exit: {' | '.join(reasons)} (pnl={pnl:+.2f}%)")

    if action=="LONG_ENTRY":
        code,msg=place_swap("buy")
        if code==200: state["position"]="LONG"; state["entry_price"]=price; state["bars_held"]=0; state["peak_price"]=price
        print(f"{'✅' if code==200 else '❌'} LONG_ENTRY → {code} | {msg[:60]}")
    elif action=="SHORT_ENTRY":
        code,msg=place_swap("sell")
        if code==200: state["position"]="SHORT"; state["entry_price"]=price; state["bars_held"]=0; state["peak_price"]=price
        print(f"{'✅' if code==200 else '❌'} SHORT_ENTRY → {code} | {msg[:60]}")
    elif action and "EXIT" in action:
        side="sell" if state["position"]=="LONG" else "buy"
        code,msg=place_swap(side)
        if code==200: state["position"]=None; state["entry_price"]=0; state["bars_held"]=0; state["peak_price"]=0
        print(f"{'✅' if code==200 else '❌'} EXIT_{state['position']} → {code} | {msg[:60]}")
    else:
        pos=state.get("position")
        print(f"No action | position={pos} | signal={smooth:+.3f}")

    with open(STATE_FILE,"w") as f: json.dump(state,f,indent=2)


if __name__=="__main__":
    if len(sys.argv)>1: INST_ID=sys.argv[1]
    run()
