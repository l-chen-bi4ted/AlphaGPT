# AlphaGPT — Signal Trading Framework

基于多因子共识信号 + ADX 市场环境感知的量化交易框架。
支持 OKX CEX 现货，信号驱动执行 + 网格机器人双轨并行。

```
                         ┌─────────────────────┐
                         │    TV Desktop        │
                         │  (可视化监控/CDP)     │
                         │   Pine Script 策略   │
                         │   Bollinger + ADX    │
                         └─────────┬───────────┘
                                   │ CDP (port 9222)
                         ┌─────────▼───────────┐
                         │  tradingview-mcp     │
                         │  截图/指标/图表控制  │
                         └─────────┬───────────┘
                                   │ 信号参考
         ┌─────────────────────────┼──────────────────────┐
         │                         │                      │
  ┌──────▼──────┐          ┌───────▼──────┐       ┌──────▼──────┐
  │ signal_bot  │          │  Grid Bot   │       │   OKX CLI   │
  │ 多因子投票   │          │  BTC/ETH网格 │       │  手动操作    │
  │ ADX 环境过滤 │          │ 服务端运行   │       │             │
  │ EMA 平滑确认 │          │ 24/7 自动   │       │             │
  └──────┬──────┘          └─────────────┘       └─────────────┘
         │
  ┌──────▼──────┐
  │  okx spot   │
  │  place (CLI)│
  │  直接下单   │
  └─────────────┘
```

## 分支说明

| 分支 | 用途 |
|------|------|
| `prod` | **当前推荐** — 信号交易 + 网格 + 文档 |
| `prod-dev` | 代码审计修复版，含完整模型训练管线 |
| `prod-v3` | 旧版 RiskEngine + MarketRegime 总装版 |
| `backtest-v2` | 回测引擎 + IC reward |
| `okx_dev` | OKX CEX 适配（稳定） |
| `main` | 上游原始版本（Solana Meme 币） |

## 快速开始

### 依赖

```bash
# Python 3.10+
pip install numpy python-dotenv requests

# Node.js 18+（TV MCP Server 可选）
# OKX CLI
npm install -g @okx_ai/okx-trade-cli
```

### 配置

```bash
# 1. OKX API 凭证
cp .env.example .env
# 编辑 .env，填入 OKX API Key（实盘 + 模拟盘）

# 2. OKX CLI
okx config init
# 按提示选择站点 → 设置模拟盘 → 填入 API Key

# 3. 验证
okx --demo account balance
# 应显示模拟盘余额
```

### 运行信号机器人

```bash
python3 signal_bot.py BTC-USDT 1H
```

每小时自动运行（cron）：

```bash
# crontab -e
0 * * * * cd /path/to/project && python3 signal_bot.py BTC-USDT 1H
```

## 组件

### signal_bot.py

多因子信号引擎，每小时轮询并自动交易。

**信号流水线**：

```
OKX K线 (200根)
    │
    ├─ RET (收益率)        权重 0.35
    ├─ LIQ (流动性)        权重 0.25  ← Amihud 非流动性指标
    ├─ PRESSURE (买卖压力) 权重 0.25
    └─ FOMO (趋势偏离)     权重 0.15  ← 仅高置信度时参与
    │
    ▼
  加权投票 → 符号共识 → EMA3 平滑
    │
    ▼
  ADX 环境检测
    ├─ TRENDING → 阈值 0.5
    ├─ RANGING  → 阈值 0.7
    └─ VOLATILE → 暂停交易
    │
    ▼
  进场确认：连续 2 根 K 线信号同向
    │
    ▼
  执行：okx spot place (市价单)
```

**持仓退出门禁**（三重保护）:

| 条件 | 触发 |
|------|------|
| 信号反转 | smoothed < -threshold |
| 持仓超时 | 24 根 K 线未反转 |
| 浮盈跟踪 | 盈利 > 2% 后回落 0.5% |

### 网格机器人

OKX 服务端运行，无需本地进程：

```bash
# 创建 BTC 网格
okx --demo bot grid create \
  --instId BTC-USDT \
  --algoOrdType grid \
  --maxPx 85000 --minPx 78000 \
  --gridNum 20 --quoteSz 5000

# 查看状态
okx --demo bot grid orders --algoOrdType grid

# 停止
okx --demo bot grid stop --algoId <ID> --algoOrdType grid --instId BTC-USDT
```

### TV Desktop 可视化（可选）

需要 TradingView Desktop（付费版）+ CDP 调试端口启动：

```bash
/Applications/TradingView.app/Contents/MacOS/TradingView \
  --remote-debugging-port=9222
```

连接 MCP 工具：

```bash
git clone https://github.com/tradesdontlie/tradingview-mcp.git
cd tradingview-mcp && npm install

# 检查连接
node src/cli/index.js status

# 常用操作
node src/cli/index.js symbol BTCUSDT
node src/cli/index.js timeframe 60
node src/cli/index.js indicator add "Bollinger Bands"
node src/cli/index.js indicator add "Directional Movement"
node src/cli/index.js screenshot --region chart
```

## 配置文件

### .env

```
OKX_API_KEY=your_live_key
OKX_SECRET_KEY=your_live_secret
OKX_PASSPHRASE=your_live_passphrase
OKX_DEMO_API_KEY=your_demo_key
OKX_DEMO_SECRET_KEY=your_demo_secret
OKX_DEMO_PASSPHRASE=your_demo_passphrase
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

> ⚠️ `passphrase` 等敏感信息以明文存储在此文件中。建议设置文件权限 `chmod 600 ~/.okx/config.toml`。

## 已知问题

| 问题 | 原因 | 解决 |
|------|------|------|
| OKX API 连接超时 | macOS LibreSSL + 系统代理 | Python 请求加 `proxies={"http": None, "https": None}` |
| GBK 编码崩溃 | Windows 终端不支持 emoji | 所有输出用 ASCII 字符 |
| TV MCP 添加指标失败 | Web 版 vs Desktop 版 UI 差异 | 使用 TradingView Desktop， |
| OKX Demo 市价单不成交 | 模拟盘流动性不足 | 改用限价单（买=ask，卖=bid） |

## 免责声明

本框架仅供学习和研究目的。信号由多因子模型生成，**历史回测表现不代表未来收益**。使用前请充分测试，量化交易存在本金损失风险。
