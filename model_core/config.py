import torch
import os


class ModelConfig:
    # ─── 硬件 ─────────────────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── OKX 市场配置 ────────────────────────────────────
    INST_ID = os.getenv("OKX_INST_ID", "BTC-USDT")  # 交易对
    BAR = os.getenv("OKX_BAR", "1H")                 # K 线周期
    CANDLE_LIMIT = int(os.getenv("OKX_CANDLE_LIMIT", "2000"))  # 拉取条数

    # ─── 训练超参 ────────────────────────────────────────
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8192"))
    TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", "1000"))
    MAX_FORMULA_LEN = int(os.getenv("MAX_FORMULA_LEN", "12"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))

    # ─── CEX 费率（OKX 现货 VIP0 taker ≈ 0.1%） ─────────
    BASE_FEE = float(os.getenv("BASE_FEE", "0.001"))  # 单边手续费率
    MIN_LIQUIDITY = 0.0  # CEX 无需流动性检查

    # ─── 回测参数 ────────────────────────────────────────
    TRADE_SIZE_USD = float(os.getenv("TRADE_SIZE_USD", "1000.0"))
    SIGNAL_THRESHOLD = float(os.getenv("SIGNAL_THRESHOLD", "0.5"))  # 信号 > 此值入场

    # ─── 特征维度（与 FeatureEngineer.INPUT_DIM 一致）─────
    INPUT_DIM = 6

    # ─── 日志 ─────────────────────────────────────────────
    SAVE_DIR = os.getenv("SAVE_DIR", "./output")
    os.makedirs(SAVE_DIR, exist_ok=True)
