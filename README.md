# AlphaGPT + OKX — CEX 符号回归量化交易

基于 [AlphaGPT](https://github.com/imbue-bit/AlphaGPT) 的 OKX CEX 适配版。用强化学习在 OKX K 线数据上自动挖掘 alpha 因子公式，支持回测和实盘/模拟盘执行。

## 与原项目的差异

| | 原版 AlphaGPT | 本 Fork |
|---|---|---|
| 市场 | Solana Meme 币 | OKX CEX 现货 |
| 数据源 | PostgreSQL + Birdeye/DexScreener | OKX REST API |
| 执行层 | Jupiter DEX (Solana) | OKX REST API (HMAC) |
| 费率模型 | 0.6% (Swap+Gas+Jito) | 0.1% (VIP0 taker) |
| 依赖 | solana-py, solders, asyncpg | requests (纯 HTTP) |
| macOS | 部分兼容 | 完全兼容 |

## 快速开始

```bash
# 1. 安装
pip install -r requirements.txt

# 2. 配环境变量（可选——训练不需要 API key）
cp .env.example .env

# 3. 训练因子公式
python model_core/engine.py

# 4. 查看结果
cat output/BTCUSDT_1H_formula.json
```

## 项目结构

```
AlphaGPT/
├── okx_data.py              # OKX 行情数据拉取
├── okx_executor.py          # OKX 实盘/模拟盘交易
├── live_runner.py           # 实盘运行器
├── model_core/              # 核心模型（复用原项目）
│   ├── alphagpt.py          # Looped Transformer 模型
│   ├── engine.py            # RL 训练引擎
│   ├── backtest.py          # CEX 回测引擎
│   ├── vm.py                # 后缀表达式虚拟机
│   ├── ops.py               # 操作符集
│   ├── factors.py           # 特征工程
│   └── config.py            # 配置
├── test_e2e.py              # 端到端测试
└── train_quick.py           # 快速训练脚本
```

## 配置

通过 `.env` 文件或环境变量：

```bash
OKX_INST_ID=BTC-USDT      # 交易对
OKX_BAR=1H                # K 线周期 (1m/5m/15m/1H/4H/1D)
OKX_CANDLE_LIMIT=2000     # 拉取条数

# 以下仅实盘需要
OKX_API_KEY=your_key
OKX_SECRET_KEY=your_secret
OKX_PASSPHRASE=your_passphrase
```

## 实盘/模拟盘

```bash
# 模拟盘（安全，不涉及真实资金）
python live_runner.py          # 依赖 formula.json

# 实盘（需配 API key）
DEMO=false python live_runner.py
```

## 模型原理

AlphaGPT 用 REINFORCE 强化学习训练一个小型 Looped Transformer，让它生成后缀表达式（逆波兰表示法）公式。每个公式被 StackVM 执行后，由 CEXBacktest 在历史 K 线上评估，回报作为奖励信号。

核心创新：
- **Looped Transformer**: 每层循环 3 次，增强表示能力
- **QK-Norm + RMSNorm + SwiGLU**: 现代化 Transformer 组件
- **Newton-Schulz LoRD**: 低秩正则化防止过拟合
- **MTP Head**: 多任务池化头

## License

Apache 2.0（继承原项目）
