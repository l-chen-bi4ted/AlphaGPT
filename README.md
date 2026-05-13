# AlphaGPT — Signal Trading Framework

基于多因子共识信号 + ADX 市场环境感知的量化交易框架。

```
                         ┌─────────────────────┐
                         │    TV Desktop        │
                         │  (可视化监控/CDP)     │
                         └─────────┬───────────┘
                                   │ 信号参考
         ┌─────────────────────────┼──────────────────────┐
         │          现货池         │        合约池         │
  ┌──────▼──────┐         ┌───────▼───────┐      ┌───────▼──────┐
  │ Grid Bot    │         │ signal_bot    │      │ swap_bot    │
  │ BTC/ETH网格 │         │ spot buy only │      │ long+short  │
  │ 服务端运行   │         │ 25% 资金      │      │ 35% 资金    │
  │ 40% 资金    │         │              │      │ 3x 杠杆     │
  └─────────────┘         └──────────────┘      └──────────────┘
```

## 分支说明

| 分支 | 用途 |
|------|------|
| `prod` | **推荐** — 信号交易 + 网格 + 合约三合一 |
| `prod-dev` | 代码审计修复版 |
| `prod-v3` | 旧版 RiskEngine |
| `main` | 上游原始版本（Solana Meme） |

## 快速开始

```bash
pip install numpy python-dotenv requests

# OKX CLI
npm install -g @okx_ai/okx-trade-cli

# 配置
cp .env.example .env   # 填入 API Key
okx config init        # 按提示配置

# 验证
okx --demo account balance
```

## 三池架构

| 池 | 脚本 | 标的 | 方向 | 资金占比 |
|---|------|------|------|:-------:|
| 网格池 | `okx bot grid` | BTC, ETH 现货 | 低买高卖 | 40% |
| 现货信号池 | `signal_bot.py` | BTC 现货 | 做多 | 25% |
| 合约信号池 | `signal_bot_swap.py` | BTC 永续合约 | 做多+做空 | 35% (3x) |

### 网格池

```bash
okx --demo bot grid create \
  --instId BTC-USDT --algoOrdType grid \
  --maxPx 85000 --minPx 78000 --gridNum 20 --quoteSz 5000
```

自动震荡市吃波动，OKX 服务端运行，零本地进程。

### 现货信号池

```bash
python3 signal_bot.py BTC-USDT 1H
```

| 阶段 | 逻辑 |
|------|------|
| 信号 | RET + LIQ + PRESSURE + FOMO 共识投票 |
| 平滑 | EMA3 滤波 |
| 环境 | ADX / TRENDING / RANGING / VOLATILE |
| 进场 | 连续 2 根同向 + 超阈值 |
| 出场 | 反转 / 24H 超时 / 浮盈回落 0.5% |

### 合约信号池

```bash
python3 signal_bot_swap.py BTC-USDT-SWAP
```

与现货池共享信号逻辑，但支持双向：

| 方向 | 信号条件 |
|------|---------|
| LONG | signal > +threshold |
| SHORT | signal < -threshold |

参数：

| 参数 | 值 |
|------|-----|
| 杠杆 | 3x |
| 单次 | 1 合约 (0.01 BTC) |
| 保证金 | 逐仓 cross |

### 离线运行（网络受限环境）

```bash
python3 signal_bot_offline.py data_cache/BTCUSDT_1H.csv
```

从 CSV 缓存读数据计算信号，无需访问 OKX API。

## TV Desktop 可视化（可选）

TradingView Desktop + CDP 调试端口：

```bash
/Applications/TradingView.app/Contents/MacOS/TradingView \
  --remote-debugging-port=9222
```

```bash
git clone https://github.com/tradesdontlie/tradingview-mcp.git
cd tradingview-mcp && npm install
node src/cli/index.js status
node src/cli/index.js symbol BTCUSDT
node src/cli/index.js indicator add "Bollinger Bands"
node src/cli/index.js indicator add "Directional Movement"
node src/cli/index.js screenshot --region chart
```

## 配置

### .env

```
OKX_DEMO_API_KEY=xxx
OKX_DEMO_SECRET_KEY=xxx
OKX_DEMO_PASSPHRASE=xxx
```

### ~/.okx/config.toml

```toml
default_profile = "demo"
[profiles.demo]
site = "global"
demo = true
api_key = "..."
secret_key = "..."
passphrase = "..."
```

## 注意事项

- 模拟盘合约无爆仓风险，但养成风控习惯
- 信号脚本已包含三重出场保护（反转/超时/浮盈跟踪）
- 网格与信号池资金隔离，互不干扰
- 实盘调低杠杆至 1x，资金分配按实际风险承受调整
