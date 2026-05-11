# AlphaGPT (prod-v3)

基于强化学习的符号回归因子挖掘引擎 — **工业级风控装甲版**。

## 分支

| 分支 | 用途 |
|------|------|
| `main` | 上游原始版本（Solana Meme 币） |
| `okx_dev` | OKX CEX 适配 + 模拟盘 live runner（稳定） |
| `backtest-v2` | 回测引擎 + IC reward + 样本外验证（研究） |
| `richenlin-analysis` | richenlin fork 拆解分析 |
| **`prod-v3`** | **RiskEngine + MarketRegime 总装版（当前）** |

## 核心架构

```
OKX K 线数据 ──→ MarketRegime (ADX/ATR 环境感知)
                      ↓
              RiskEngine (四级防御状态机)
                      ↓
              v2 liquidity 公式 → 信号交易
```

### 风控三层

| 层级 | 模块 | 功能 |
|------|------|------|
| Layer 1 环境感知 | `market_regime.py` | ADX/ATR 检测 TRENDING/RANGING/VOLATILE |
| Layer 2 持仓风控 | `risk_engine.py` | 硬止损 / 追踪止损 / 分批止盈 / 时间止损 |
| Layer 3 账户保护 | `risk_engine.py` | 日回撤熔断 / 连亏降级 / 四级递进 |

## RiskEngine — 四级防御

```
NORMAL ──日回撤>5%──→ REDUCED ──日回撤>10%──→ PROTECTION ──日回撤>20%──→ EMERGENCY
 1.0x               0.5x仓位           只平仓不开           强制全平
```

| 等级 | 触发条件 | 行为 |
|------|---------|------|
| NORMAL | 默认 | 满额交易 |
| REDUCED | 日回撤 > 5% 或连亏 4 次 | 仓位减半 |
| PROTECTION | 日回撤 > 10% | 停止开仓，只允许平仓 |
| EMERGENCY | 日回撤 > 20% | 强制市价全平 |

## MarketRegime — 环境联调

| 状态 | ADX | 止损 | 追踪 | 止盈 | 仓位 |
|------|-----|------|------|------|------|
| TRENDING | > 25 | -5% | 6% | [10%, 20%] | 1.0x |
| RANGING | < 20 | -3% | 4% | [5%, 10%] | 0.7x |
| VOLATILE | ATR 飙升 | -8% | 10% | [15%, 30%] | 0.5x |

震荡市还自动提高入场阈值（0.5 → 0.7），减少假突破。

## 工具

```bash
# 模拟盘（prod-v3 装甲版）
OKX_INST_ID=BTC-USDT python live_runner.py

# 公式审计
python eval_formula.py BTC-USDT 1H --adversarial 10

# 超参扫描
python sweep.py BTC-USDT
```

## 当前公式

| 品种 | Train IC | Val IC | 公式 |
|------|---------|--------|------|
| **BTC v2 liquidity** | 0.054 | 0.041 | LIQ→MAX3→ABS×3→JUMP→ABS×3→MAX3→SIGN→ABS |

## 环境

```bash
pip install torch numpy pandas scipy requests python-dotenv tqdm loguru
```

CEX 交易需配置 `.env`（实盘 + 模拟盘双组 Key）。

## 参考

- [richenlin/AlphaGPT](https://github.com/richenlin/AlphaGPT) — RiskEngine 来源
- [no_JIT](https://github.com/imbue-bit/no_JIT) — HJI 微分博弈
- [szd5654125/AlphaGPT](https://github.com/szd5654125/AlphaGPT) — 时序算子

## 许可

Apache 2.0
