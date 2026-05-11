# AlphaGPT (backtest-v2)

基于强化学习的符号回归因子挖掘引擎，适配 OKX CEX 现货市场。

## 分支

| 分支 | 用途 |
|------|------|
| `main` | 上游原始版本（Solana Meme 币） |
| `okx_dev` | OKX CEX 适配 + 模拟盘 live runner（稳定运行） |
| `backtest-v2` | 新回测引擎 + IC reward + 样本外验证 + 风控装甲（活跃开发） |

## 核心架构

```
OKX K 线数据 → FeatureEngineer (6因子) → StackVM (RPN执行)
                                              ↓
AlphaGPT (Looped Transformer) → 采样公式token → Rank IC 评估
                                              ↓
                                    REINFORCE 梯度更新
                                              ↓
                              对抗噪声筛选 + 样本外验证
```

### 关键改进 (vs 上游)

| 改进 | 说明 |
|------|------|
| **Rank IC 训练** | reward = IC × 10，替代脆弱的回测收益驱动 |
| **样本外 split** | 时间序列 80% 训练 / 20% 验证，自动过拟合警告 |
| **对抗筛选** | 因子注入噪声，取 worst-case 得分 |
| **多维 fitness** | Sharpe + Sortino + 胜率 + 最大回撤 综合评分 |
| **复杂度 + 重复惩罚** | 唯一算子数 + 单算子重复超 4 次扣分，防 STD20×10 退化 |
| **4 个时序算子** | DELTA5 / MA20 / STD20 / TS_RANK20（szd fork） |
| **GPU 加速** | 训练数据搬上 CUDA，VM 执行 10x 加速 |
| **向量化 IC** | 批量 argsort，100x 快于 Python 循环 |
| **live_runner 风控** | 硬止损 -5% / 追踪止损 6% / 单日熔断 15% / 冷却 8h |

## 工具

```bash
# 公式审计 — 秒级 OOS 评估
python eval_formula.py BTC-USDT 1H --adversarial 10

# 超参扫描 — 3 组 × 500 步
python sweep.py BTC-USDT

# 模拟盘（v2 风控装甲）
OKX_INST_ID=BTC-USDT python live_runner.py
```

## IC 基准

| IC 值 | 评价 |
|-------|------|
| > 0.05 | 优秀 |
| 0.03 - 0.05 | 有效 |
| < 0.03 | 弱/噪声 |

## 当前最佳公式

| 品种 | Train IC | Val IC | 公式 |
|------|---------|--------|------|
| **BTC v2 liquidity** | 0.054 | **0.041** | LIQ→MAX3→ABS×3→JUMP→ABS×3→MAX3→SIGN→ABS |
| BTC v1 price | 0.020 | -0.014 | DEV 驱动（已弃用） |
| ETH | 0.031 | 0.033 | 弱有效 |
| SOL | 0.019 | 0.000 | 过拟合 |

> 新算子 sweep 产出的高 IC 公式 (0.11) 全为 STD20 退化型，Val IC 为负，已弃用。

## 风控参数 (live_runner v2)

| 参数 | 值 | 说明 |
|------|-----|------|
| 硬止损 | -5% | 亏损 5% 强制平仓 |
| 追踪止损 | -6% | 从最高点回撤 6% 平仓 |
| 单日熔断 | -15% | 单日亏损超 15% 熔断 |
| 冷却时间 | 8h | 风控触发后暂停交易 |
| 初始保护 | z-score < 30 | 信号历史不足时不买入 |

## 环境

```bash
pip install torch numpy pandas scipy requests python-dotenv tqdm loguru
```

CEX 交易需配置 `.env`：
```
OKX_API_KEY=xxx
OKX_SECRET_KEY=xxx
OKX_PASSPHRASE=xxx
OKX_DEMO_API_KEY=xxx
OKX_DEMO_SECRET_KEY=xxx
OKX_DEMO_PASSPHRASE=xxx
```

训练不需要 API key —— 使用 `data_cache/` 下的离线 CSV 数据。

## 参考

- [no_JIT](https://github.com/imbue-bit/no_JIT) — HJI 微分博弈 Uniswap V4 Hook
- [szd5654125/AlphaGPT](https://github.com/szd5654125/AlphaGPT) — A 股适配 + 时序算子
- [richenlin/AlphaGPT](https://github.com/richenlin/AlphaGPT) — Docker + 告警 + 风控架构

## 许可

Apache 2.0
