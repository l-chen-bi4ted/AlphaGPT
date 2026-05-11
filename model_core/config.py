import torch
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """实例化配置，支持并行隔离。"""

    # ─── 硬件 ─────────────────────────────────────────────
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))

    # ─── OKX 市场配置 ────────────────────────────────────
    inst_id: str = field(default_factory=lambda: os.getenv("OKX_INST_ID", "BTC-USDT"))
    bar: str = field(default_factory=lambda: os.getenv("OKX_BAR", "1H"))
    candle_limit: int = field(default_factory=lambda: int(os.getenv("OKX_CANDLE_LIMIT", "2000")))

    # ─── 训练超参 ────────────────────────────────────────
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "8192")))
    train_steps: int = field(default_factory=lambda: int(os.getenv("TRAIN_STEPS", "1000")))
    max_formula_len: int = field(default_factory=lambda: int(os.getenv("MAX_FORMULA_LEN", "12")))
    learning_rate: float = field(default_factory=lambda: float(os.getenv("LEARNING_RATE", "1e-3")))

    # ─── CEX 费率（OKX 现货 VIP0 taker ≈ 0.1%） ─────────
    base_fee: float = field(default_factory=lambda: float(os.getenv("BASE_FEE", "0.001")))
    min_liquidity: float = 0.0

    # ─── 回测参数 ────────────────────────────────────────
    trade_size_usd: float = field(default_factory=lambda: float(os.getenv("TRADE_SIZE_USD", "1000.0")))
    signal_threshold: float = field(default_factory=lambda: float(os.getenv("SIGNAL_THRESHOLD", "0.5")))

    # ─── 滑点与冲击 ──────────────────────────────────────
    slippage_bps: float = field(default_factory=lambda: float(os.getenv("SLIPPAGE_BPS", "5.0")))  # 万分之五
    latency_bars: int = field(default_factory=lambda: int(os.getenv("LATENCY_BARS", "1")))  # 信号→执行延迟 K 线数

    # ─── v2 对抗回测 ────────────────────────────────────
    adversarial_trials: int = field(default_factory=lambda: int(os.getenv("ADVERSARIAL_TRIALS", "0")))
    adversarial_noise: float = field(default_factory=lambda: float(os.getenv("ADVERSARIAL_NOISE", "0.02")))
    multi_dim_fitness: bool = field(default_factory=lambda: os.getenv("MULTI_DIM_FITNESS", "true").lower() in ("true", "1"))

    # ─── 特征维度（与 FeatureEngineer.INPUT_DIM 一致）─────
    input_dim: int = 6

    # ─── 日志 ─────────────────────────────────────────────
    save_dir: str = field(default_factory=lambda: os.getenv("SAVE_DIR", "./output"))

    # ─── 内部状态 ─────────────────────────────────────────
    _save_dir_created: bool = field(default=False, repr=False)

    def __post_init__(self):
        if not self._save_dir_created:
            os.makedirs(self.save_dir, exist_ok=True)
            self._save_dir_created = True

    def to_dict(self) -> dict:
        return {
            k: str(v) if isinstance(v, torch.device) else v
            for k, v in self.__dict__.items() if not k.startswith("_")
        }

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls()


# 全局默认实例（向后兼容）
default_config = ModelConfig()
