import torch

# ─── 基础工具 ──────────────────────────────────

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    """滞后 d 期。d=0 返回原值，d>0 前置填充 0。"""
    if d == 0:
        return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

@torch.jit.script
def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    """d 期差分：x - delay(x, d)。"""
    return x - _ts_delay(x, d)

@torch.jit.script
def _ts_decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    """线性衰减加权移动平均（窗口 d）。权重 1,2,...,d 归一化。
    前 d-1 根 K 线填 0（避免未来函数）。"""
    if d <= 1:
        return x
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)                     # [B, d-1+T]
    windows = x_pad.unfold(1, d, 1)                        # [B, T, d]
    w = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    w = w / w.sum()
    return (windows * w).sum(dim=-1)                       # [B, T]

@torch.jit.script
def _ts_zscore(x: torch.Tensor, d: int) -> torch.Tensor:
    """滚动 z-score（窗口 d）。前 d-1 根填 0。"""
    if d <= 1:
        return torch.zeros_like(x)
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)                        # [B, T, d]
    mean = windows.mean(dim=-1)
    std = windows.std(dim=-1) + 1e-6
    return (x - mean) / std

@torch.jit.script
def _ts_percentile_rank(x: torch.Tensor, d: int) -> torch.Tensor:
    """滚动百分位秩（窗口 d）。返回值 0~1，前 d-1 根填 0.5。
    TS_RANK20(x) = 过去20根中 x 排在什么位置。"""
    if d <= 1:
        return torch.full_like(x, 0.5)
    B, T = x.shape
    pad = torch.zeros((B, d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)                     # [B, d-1+T]
    windows = x_pad.unfold(1, d, 1)                        # [B, T, d]
    # 计算每帧中当前值在窗口内的秩 (0~1)
    current = x.unsqueeze(-1)                               # [B, T, 1]
    rank = (windows < current).float().sum(dim=-1)          # [B, T]
    max_rank = windows.shape[-1] - 1                       # d-1
    # 前 d-1 帧 windows 含大量 0，rank 偏低 → 统一返回 0.5
    mask = torch.zeros(T, device=x.device)
    mask[d-1:] = 1.0
    result = rank / float(max(max_rank, 1))
    # 窗口不足时用 0.5 填充
    return mask.unsqueeze(0) * result + (1 - mask.unsqueeze(0)) * 0.5

# ─── 原始算子 ──────────────────────────────────

@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

@torch.jit.script
def _op_jump(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)

@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)


OPS_CONFIG = [
    # 基础二元
    ('ADD',  lambda x, y: x + y, 2),
    ('SUB',  lambda x, y: x - y, 2),
    ('MUL',  lambda x, y: x * y, 2),
    ('DIV',  lambda x, y: x / (y + 1e-6), 2),
    # 基础一元
    ('NEG',  lambda x: -x, 1),
    ('ABS',  torch.abs, 1),
    ('SIGN', torch.sign, 1),
    # 高级
    ('GATE',  _op_gate, 3),
    ('JUMP',  _op_jump, 1),
    ('DECAY', _op_decay, 1),
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('MAX3',   lambda x: torch.max(x, torch.max(_ts_delay(x, 1), _ts_delay(x, 2))), 1),
    # ── v2: 时间序列纵深算子（szd fork + Manus review）──
    ('DELTA5',    lambda x: _ts_delta(x, 5), 1),            # 5期中短期动量
    ('MA20',      lambda x: _ts_decay_linear(x, 20), 1),    # 20期线性衰减均线
    ('STD20',     lambda x: _ts_zscore(x, 20), 1),          # 20期z-score（异常检测）
    ('TS_RANK20', lambda x: _ts_percentile_rank(x, 20), 1), # 20期百分位秩（鲁棒归一化）
]
