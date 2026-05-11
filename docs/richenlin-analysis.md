# richenlin/AlphaGPT 深度拆解

## 概览

richenlin fork 将上游的 Solana Meme 币原型重构成了一个**准生产级全栈量化系统**。
~10,000 行 Python + Docker 部署 + 测试覆盖 + 风控引擎 + API 服务。

## 架构分层

```
配置层: config.yaml, config/settings.py
   ↓
数据层: data_pipeline/ (Binance/OKX/CoinGecko/Birdeye + WebSocket)
   ↓
执行层: execution/ (多交易所 router, 滑点预测, 限价/市价)
   ↓
策略层: strategy_manager/ (runner, risk_engine, portfolio, HFT scalper)
   ↓
学习层: rl/ (PPO在线学习, GAE, state encoder, reward calculator)
   ↓
服务层: api/ + dashboard/ + metrics/ (Prometheus)
```

## 和我们直接相关的模块

### 1. 风控引擎 (risk_engine.py, 707行)

**三层架构**:

| 层级 | 检查内容 | 触发动作 |
|------|---------|---------|
| Layer 1 交易前 | 流动性、蜜罐检测、仓位限制、最大持仓数 | 拒绝或调小仓位 |
| Layer 2 持仓监控 | 止损、止盈、追踪止损、时间止损 | 平仓 |
| Layer 3 账户保护 | 日回撤、连亏暂停、紧急平仓 | 全局熔断 |

**RiskLevel 递进机制**:
```
NORMAL → REDUCED → PROTECTION → EMERGENCY
  ↓         ↓           ↓            ↓
 正常      减仓30%    只平仓不开    全部平仓
```

触发条件:
- 日回撤超过阈值 → REDUCED
- 连续亏损 3 次 → PROTECTION
- 账户总回撤 > 40% → EMERGENCY

### 2. 市场状态检测 (market_regime.py + market_state_analyzer.py)

六种市场状态:
- TRENDING_UP / TRENDING_DOWN: 趋势（ADX > 25）
- RANGING: 震荡（ADX < 20）
- HIGH_VOLATILITY: 高波动（ATR > 阈值）
- LOW_VOLATILITY: 低波动
- BREAKOUT: 突破

各状态对应不同参数: 趋势用宽止损，震荡用紧止损，高波动减仓。

### 3. 模式管理器 (mode_manager.py)

```python
# 自动模式切换：$1000 为界
if total_value >= 1000:
    return BALANCED   # 多持仓、紧止损、低杠杆
else:
    return AGGRESSIVE  # 少持仓、宽止损、高杠杆
```

太简单——纯资金量判断，无市场状态感知。

### 4. 追踪止损实现 (aggressive_strategy.py + runner.py)

```python
# aggressive_strategy: 基于当前价的百分比追踪
def get_trailing_stop_price(current_price):
    return current_price * (1 - trailing_pct)  # long

# runner.py: 需要先盈利一定幅度才激活
max_gain = (pos.highest_price - pos.entry_price) / pos.entry_price
drawdown = (pos.highest_price - current_price) / pos.highest_price
if max_gain > TRAILING_ACTIVATION and drawdown > TRAILING_DROP:
    # 触发追踪止损
```

和我们已实现的一致。我们多了一个"冷却"概念，richenlin 没有。

### 5. 主循环 (runner.py, 355行)

```python
async def run_loop():
    while True:
        # 1. 全局熔断检查
        is_allowed, reason = self.risk.check_account_risk()
        
        # 2. 每 15 分钟同步数据管道
        if time.time() - self.last_scan > 900:
            data_mgr.pipeline_sync_daily()
        
        # 3. 加载数据 → 扫描入场 → 监控持仓
        loader.load_data(limit_tokens=300)
        scan_for_entries()
        monitor_positions()
        
        # 4. 循环间隔 60 秒（不是 1 小时！）
        sleep(60)
```

**和我们的区别**: 它是高频轮询（60秒一次），我们是 1H K 线对齐（3600秒）。它一次扫描 300 个 token，我们只盯一个。

### 6. HFT 模块 (strategy_manager/hft/)

- **market_regime_detector.py**: ADX + ATR 检测市场状态
- **scalper.py**: 剥头皮策略（0.8% 止盈，0.4% 止损，最长持仓 30 分钟）

### 7. RL 在线学习 (rl/)

richenlin 加了 PPO 在线学习框架:
- PPO network（Actor-Critic）
- GAE 优势估计
- Prioritized Replay Buffer
- Online Learner（实时样本训练）
- Model Version Manager（版本控制）
- Reward Calculator（多维 reward）

**和我们**: 我们的 REINFORCE 训练是离线的，richenlin 试图做在线持续学习。

### 8. 安全/运维

- **wallets/**: 冷热钱包管理 + 多签
- **security/**: 密钥加密 + 审计日志
- **alerts/**: Telegram 告警机器人
- **metrics/**: Prometheus 指标（350行）
- **Docker**: docker-compose.yml + Dockerfile
- **tests/**: 20+ 测试文件，包含 mock exchange

## 可以拿的

| 模块 | 价值 | 工程复杂度 |
|------|------|-----------|
| RiskLevel 递进机制 | 高——熔断后渐进恢复 | 中 |
| 市场状态检测 | 高——动态切换参数 | 中 |
| HFT scalper | 中——需要高频数据 | 高 |
| 在线 PPO 学习 | 中——概念好但太重 | 高 |
| Telegram 告警 | 低——独立模块 | 低 |
| Prometheus 指标 | 低——运维需求 | 低 |

## 可以留的

| 模块 | 理由 |
|------|------|
| 冷热钱包/多签 | 单地址量太小 |
| JIT/三明治防御 | 现货不需要 |
| Solana 执行层 | 我们已经切 OKX |
| Meme 币数据管道 | 已弃用 |

## 建议下一步

1. **拿 RiskLevel 递进**: NORMAL → REDUCED → PROTECTION → EMERGENCY，替代我们的 hard cooldown
2. **拿市场状态检测**: ADX + ATR 动态切参数，替代手动 RISK_MODE
3. **Telegram 告警**: 轻量独立，cooldown/熔断时推送通知
