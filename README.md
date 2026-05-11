# AlphaGPT (prod-dev)

基于强化学习的符号回归因子挖掘引擎，适配 OKX CEX 现货市场。

prod-dev 是从 prod-v3 经过完整代码审计后的修复分支，主要改动：
- 消除数据层面的未来函数（`robust_norm`、`target_ret`）
- 回测加入真实摩擦（滑点、时滞、手续费）
- RiskEngine 与 live_runner 闭环联动
- OKX 执行器接入成交回报（`get_order` / `wait_for_fill`）
- 新增 ExhaustiveOracle 穷举搜索 + RL 预训练蒸馏
- GPU 训练优化（混合精度、torch.compile、梯度累积、可配置模型容量）

## 文件结构

```
.
├── live_runner.py          # 实盘入口（风控闭环 + 成交核对）
├── okx_executor.py         # OKX API 执行器（HMAC-SHA256 + 成交回报）
├── okx_data.py             # OKX K 线数据加载（含元数据校验）
├── fetch_cache.py          # 离线数据拉取（带 SHA256 元数据）
├── eval_formula.py         # 公式审计（对抗评估 + 阈值扫描）
├── sweep.py                # 超参扫描
├── train_quick.py          # 快速训练入口
├── test_e2e.py             # 端到端测试
├── model_core/
│   ├── alphagpt.py         # Transformer 策略网络（可配置 d_model/n_layer）
│   ├── engine.py           # RL 训练引擎（REINFORCE + Oracle 预训练）
│   ├── oracle.py           # 穷举搜索（finishable pruning + top-k heap）
│   ├── backtest.py         # CEX 回测（滑点 + 时滞 + 真实手续费）
│   ├── risk_engine.py      # 四级风控状态机（成交回报驱动）
│   ├── market_regime.py    # 市场状态检测（ADX/ATR，无未来函数）
│   ├── vm.py               # StackVM 公式执行（nan/inf 审计日志）
│   ├── ops.py              # 算子配置（16 个，含时序算子）
│   ├── factors.py          # 特征工程（滚动 robust_norm，无未来函数）
│   ├── search_space.py     # 搜索空间分析
│   └── config.py           # 实例化配置（dataclass，支持并行隔离）
├── scripts/
│   ├── train_gpu.py        # GPU 一键训练脚本
│   └── benchmark_oracle.py # Oracle 基准测试 + IC 景观图
└── docs/                   # richenlin 拆解分析文档
```

## 快速开始

### 环境安装

```bash
git clone -b prod-dev https://github.com/l-chen-bi4ted/AlphaGPT.git
cd AlphaGPT
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 填入 OKX API Key
```

### 拉取离线数据

```bash
python fetch_cache.py
# 输出: data_cache/BTCUSDT_1H.csv + .meta.json
```

### 端到端测试

```bash
python test_e2e.py
# 50 步快速训练，验证数据流 + 模型 + VM + 回测链路
```

## 训练

### 基础训练

```bash
python scripts/train_gpu.py
```

### Oracle 预训练 + RL 微调

```bash
D_MODEL=128 N_LAYER=4 BATCH_SIZE=65536 TRAIN_STEPS=5000 \
    python scripts/train_gpu.py --pretrain-oracle --inst-id ETH-USDT
```

### 仅 Oracle 穷举基准（无模型训练）

```bash
python scripts/benchmark_oracle.py BTC-USDT --max-len 8 --topk 50 --plot
```

### 超参扫描

```bash
python sweep.py BTC-USDT
```

### 环境变量（训练调参）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `D_MODEL` | 64 | 模型维度，GPU 训练建议 128-256 |
| `N_LAYER` | 2 | Transformer 层数，建议 4 |
| `N_HEAD` | 4 | 注意力头数，需整除 d_model |
| `BATCH_SIZE` | 8192 | 每步采样公式数，GPU 建议 65536-131072 |
| `GRAD_ACCUM_STEPS` | 1 | 梯度累积步数 |
| `TRAIN_STEPS` | 1000 | 训练总步数 |
| `LEARNING_RATE` | 1e-3 | AdamW 学习率 |
| `USE_AMP` | true | 混合精度训练（FP16/BF16） |
| `COMPILE_MODEL` | true | torch.compile（PyTorch 2.0+） |
| `MAX_FORMULA_LEN` | 12 | 最大公式长度 |
| `SLIPPAGE_BPS` | 5.0 | 回测滑点（万分之五） |
| `LATENCY_BARS` | 1 | 信号到执行的 K 线延迟 |

## Oracle 穷举搜索

`model_core/oracle.py` 提供带 `finishable pruning` 的穷举搜索，用于：

1. **建立 RL 性能上界**：验证模型搜索质量是否接近全局最优
2. **生成预训练数据**：top-k 公式蒸馏到 AlphaGPT，降低 RL 收敛难度
3. **IC 景观分析**：绘制公式长度/算子数 vs IC 的分布图

### 用法

```python
from model_core.oracle import ExhaustiveOracle
from model_core.config import ModelConfig
from okx_data import OKXDataLoader

config = ModelConfig()
loader = OKXDataLoader("BTC-USDT", "1H", 2000, config=config)
loader.load_data()

oracle = ExhaustiveOracle(config, max_len=6, ops_subset="basic")
results = oracle.search(loader, topk=20)

for r in results[:5]:
    print(f"IC={r.ic:.4f} ValIC={r.val_ic:.4f} Formula={r.formula}")
```

### 算子子集

| 子集 | 算子 | 搜索空间（L=6） |
|------|------|----------------|
| `basic` | ADD,SUB,MUL,DIV,NEG,ABS,SIGN | ~10^5 |
| `time` | + DELAY1,DECAY,MAX3 | ~10^6 |
| `ts` | + DELTA5,MA20,STD20,TS_RANK20 | ~10^7 |
| `all` | 全部 16 个 | ~10^8 |

建议 L≤6 用 `basic/time`，L=8 用 `ts`，L>8 仅用于研究目的。

## 实盘入口

```bash
# 模拟盘
OKX_DEMO_API_KEY=xxx OKX_DEMO_SECRET_KEY=xxx OKX_DEMO_PASSPHRASE=xxx \
    python live_runner.py BTC-USDT --demo

# 实盘（需严格风控配置）
python live_runner.py BTC-USDT
```

`live_runner.py` 运行时联动 `RiskEngine`：
- 每次下单前检查 `risk_engine.can_trade()`
- 成交后调用 `record_trade()` 更新日回撤/连亏计数
- 日回撤超阈值自动触发仓位缩放或暂停
- 信号→限价单/市价止损单，下单后查询成交回报确认

## 核心修复（相比 prod-v3）

| 问题 | 修复 |
|------|------|
| 未来函数 | `robust_norm` 改为滚动窗口；`target_ret` 修正对齐；`MarketRegime` 消除 `np.roll` |
| 回测零摩擦 | 加入 `slippage_bps`、`latency_bars`、真实手续费扣除 |
| 风控悬空 | `live_runner` 闭环调用 `RiskEngine.can_trade()` / `record_trade()` |
| 成交开环 | `okx_executor` 新增 `get_order()` / `wait_for_fill()` 成交回报 |
| 配置污染 | `ModelConfig` 改为 `dataclass` 实例化，支持多进程隔离 |
| 模型容量 | 从硬编码 64/2/4 改为从 config 读取，支持 128-256 维度 |
| 训练效率 | 混合精度 + torch.compile + 梯度累积 |
| 搜索质量 | Oracle 预训练蒸馏，RL 不再从零随机搜索 |

## 参考

- [szd5654125/AlphaGPT](https://github.com/szd5654125/AlphaGPT) — finishable pruning oracle
- [richenlin/AlphaGPT](https://github.com/richenlin/AlphaGPT) — RiskEngine 参考实现
- [no_JIT](https://github.com/imbue-bit/no_JIT) — HJI 微分博弈

## 许可

Apache 2.0
