#!/usr/bin/env python3
"""Quick training script — 500 steps ETH-USDT 1H"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_core.config import ModelConfig
ModelConfig.TRAIN_STEPS = 500
ModelConfig.BATCH_SIZE = 4096
ModelConfig.CANDLE_LIMIT = 2000

from model_core.engine import AlphaEngine
engine = AlphaEngine(inst_id="ETH-USDT", bar="1H", use_lord_regularization=False)
engine.train()
