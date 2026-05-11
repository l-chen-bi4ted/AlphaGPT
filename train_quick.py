#!/usr/bin/env python3
"""Quick training script — 500 steps BTC-USDT 1H"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_core.config import ModelConfig

config = ModelConfig()
config.train_steps = 500
config.batch_size = 4096
config.candle_limit = 2000

from model_core.engine import AlphaEngine
engine = AlphaEngine(config=config, inst_id="BTC-USDT", bar="1H", use_lord_regularization=False)
engine.train()
