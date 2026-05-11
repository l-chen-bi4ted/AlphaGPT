import torch
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """实例化配置，支持并行隔离 + GPU 性能调优。"""

    # ─── 硬件 ─────────────────────────────────────────────
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))

    # ─── GPU 性能调优（3090 24G 专用）──────────────────────
    use_amp: bool = field(default_factory=lambda: os.getenv("USE_AMP", "true").lower() in ("true", "1"))
    """混合精度训练 (torch.cuda.amp)，3090 上速度提升 ~1.5x，显存减半。"""

    compile_model: bool = field(default_factory=lambda: os.getenv("COMPILE_MODEL", "true").lower() in ("true", "1"))
    """torch.compile (PyTorch 2.0+)，3090 上速度提升 ~20-40%。"""

    grad_accum_steps: int = field(default_factory=lambda: int(os.getenv("GRAD_ACCUM_STEPS", "1")))
    """梯度累积步数。effective_batch = batch_size * grad_accum_steps。"""

    # ─── 模型容量（3090 24G 可支撑 d_model=256, n_layer=4, n_head=8）──
    d_model: int = field(default_factory=lambda: int(os.getenv("D_MODEL", "64")))
    """模型维度。小内存卡保持 64，3090 24G 建议 128-256。"""

    n_layer: int = field(default_factory=lambda: int(os.getenv("N_LAYER", "2")))
    """Transformer 层数。3090 建议 4。"""

    n_head: int = field(default_factory=lambda: int(os.getenv("N_HEAD", "4")))
    """注意力头数，必须整除 d_model。"""

    dim_feedforward: int = field(default_factory=lambda: int(os.getenv("DIM_FEEDFORWARD", "128")))
    """FFN 中间层维度。默认 2*d_model，3090 可设 512-1024。"""

    dropout: float = field(default_factory=lambda: float(os.getenv("DROPOUT", "0.1")))

    # ─── OKX 市场配置 ────────────────────────────────────
    inst_id: str = field(default_factory=lambda: os.getenv("OKX_INST_ID", "BTC-USDT"))
    bar: str = field(default_factory=lambda: os.getenv("OKX_BAR", "1H"))
    candle_limit: int = field(default_factory=lambda: int(os.getenv("OKX_CANDLE_LIMIT", "2000")))

    # ─── 训练超参（3090 24G 建议 batch_size=65536）────────
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "8192")))
    train_steps: int = field(default_factory=lambda: int(os.getenv("TRAIN_STEPS", "1000")))
    max_formula_len: int = field(default_factory=lambda: int(os.getenv("MAX_FORMULA_LEN", "12")))
    learning_rate: float = field(default_factory=lambda: float(os.getenv("LEARNING_RATE", "1e-3")))

    # ─── CEX 费率 ─────────────────────────────────────────
    base_fee: float = field(default_factory=lambda: float(os.getenv("BASE_FEE", "0.001")))
    min_liquidity: float = 0.0

    # ─── 回测参数 ────────────────────────────────────────
    trade_size_usd: float = field(default_factory=lambda: float(os.getenv("TRADE_SIZE_USD", "1000.0")))
    signal_threshold: float = field(default_factory=lambda: float(os.getenv("SIGNAL_THRESHOLD", "0.5")))

    # ─── 滑点与冲击 ──────────────────────────────────────
    slippage_bps: float = field(default_factory=lambda: float(os.getenv("SLIPPAGE_BPS", "5.0")))
    latency_bars: int = field(default_factory=lambda: int(os.getenv("LATENCY_BARS", "1")))

    # ─── 对抗回测 ────────────────────────────────────────
    adversarial_trials: int = field(default_factory=lambda: int(os.getenv("ADVERSARIAL_TRIALS", "0")))
    adversarial_noise: float = field(default_factory=lambda: float(os.getenv("ADVERSARIAL_NOISE", "0.02")))
    multi_dim_fitness: bool = field(default_factory=lambda: os.getenv("MULTI_DIM_FITNESS", "true").lower() in ("true", "1"))

    # ─── 特征维度 ─────────────────────────────────────────
    input_dim: int = 6

    # ─── 日志 ─────────────────────────────────────────────
    save_dir: str = field(default_factory=lambda: os.getenv("SAVE_DIR", "./output"))

    # ─── 内部状态 ─────────────────────────────────────────
    _save_dir_created: bool = field(default=False, repr=False)

    def __post_init__(self):
        if not self._save_dir_created:
            os.makedirs(self.save_dir, exist_ok=True)
            self._save_dir_created = True
        # 自动调整 feedforward 为 d_model 的整数倍
        if self.dim_feedforward < self.d_model:
            self.dim_feedforward = self.d_model * 2
        # 确保 n_head 整除 d_model
        if self.d_model % self.n_head != 0:
            self.n_head = max(1, self.d_model // 64)
            self.n_head = self.n_head if self.d_model % self.n_head == 0 else 1

    def to_dict(self) -> dict:
        return {
            k: str(v) if isinstance(v, torch.device) else v
            for k, v in self.__dict__.items() if not k.startswith("_")
        }

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls()


# 全局默认实例
default_config = ModelConfig()
