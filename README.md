# AlphaGPT (backtest-v2)

基于强化学习的符号回归因子挖掘引擎，适配 OKX CEX 现货市场。

## 分支

| 分支 | 用途 |
|------|------|
| `main` | 上游原始版本（Solana Meme 币） |
| `okx_dev` | OKX CEX 适配 + 模拟盘 live runner（稳定运行） |
| `backtest-v2` | 新回测引擎 + IC reward + 样本外验证（活跃开发） |

## 核心架构 (v2)

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
| **复杂度惩罚** | 奥卡姆剃刀：唯一算子 > 5 种开始扣分 |
| **HJI Regret** | 与完美预知策略的机会成本差距 |

## 工具

```bash
# 公式审计 — 秒级 OOS 评估
python eval_formula.py BTC-USDT 1H --adversarial 10

# 超参扫描 — 3 组 × 500 步，~1 小时/组
python sweep.py BTC-USDT

# 单品种深度训练
python train_quick.py
```

## IC 基准

| IC 值 | 评价 |
|-------|------|
| > 0.05 | 优秀 |
| 0.03 - 0.05 | 有效 |
| < 0.03 | 弱/噪声 |

## 环境

```bash
pip install torch numpy pandas scipy requests python-dotenv tqdm loguru
```

CEX 交易需配置 `.env`：
```
OKX_API_KEY=xxx
OKX_SECRET_KEY=xxx
OKX_PASSPHRASE=xxx
OKX_DEMO_API_KEY=xxx   # 模拟盘（可选）
OKX_DEMO_SECRET_KEY=xxx
OKX_DEMO_PASSPHRASE=xxx
```

训练不需要 API key —— 使用 `data_cache/` 下的离线 CSV 数据。

## 当前公式

| 品种 | Train IC | Val IC | 训练方式 |
|------|---------|--------|----------|
| BTC v2 (liquidity) | 0.054 | 0.041 | 对抗回测 |
| BTC v1 (price) | 0.020 | -0.014 | 旧版回测 |
| ETH | 0.031 | 0.033 | 旧版回测 |
| SOL | 0.019 | 0.000 | 旧版回测 |

## 许可

Apache 2.0
