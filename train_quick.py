#!/usr/bin/env python3
"""Quick training script — 500 steps BTC-USDT 1H"""
import sys
sys.path.insert(0, "/tmp/AlphaGPT-fork")

from model_core.config import ModelConfig
ModelConfig.TRAIN_STEPS = 500
ModelConfig.BATCH_SIZE = 4096
ModelConfig.CANDLE_LIMIT = 2000

from model_core.engine import AlphaEngine
engine = AlphaEngine(use_lord_regularization=False)
engine.train()
