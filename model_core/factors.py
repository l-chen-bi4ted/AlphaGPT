import torch
import torch.nn as nn


class RMSNormFactor(nn.Module):
    """RMSNorm for factor normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def _shift_prev(t):
    """前向偏移：prev[t] = t[t-1]，首元素置 0。无未来函数。"""
    out = torch.zeros_like(t)
    out[:, 1:] = t[:, :-1]
    return out


def rolling_robust_norm(t, window=120):
    """
    滚动 robust normalization，仅使用 [t-window+1, t] 的历史数据。
    无未来函数。
    """
    N, T = t.shape
    pad = torch.zeros((N, window - 1), device=t.device)
    t_pad = torch.cat([pad, t], dim=1)           # [N, T+window-1]
    windows = t_pad.unfold(1, window, 1)          # [N, T, window]
    median = windows.median(dim=-1).values        # [N, T]
    mad = (windows - median.unsqueeze(-1)).abs().median(dim=-1).values + 1e-6
    return torch.clamp((t - median) / mad, -5.0, 5.0)


class MemeIndicators:
    @staticmethod
    def liquidity_health(liquidity, fdv):
        ratio = liquidity / (fdv + 1e-6)
        return torch.clamp(ratio * 4.0, 0.0, 1.0)

    @staticmethod
    def buy_sell_imbalance(close, open_, high, low):
        range_hl = high - low + 1e-9
        body = close - open_
        strength = body / range_hl
        return torch.tanh(strength * 3.0)

    @staticmethod
    def fomo_acceleration(volume, window=5):
        vol_prev = _shift_prev(volume)
        vol_chg = (volume - vol_prev) / (vol_prev + 1.0)
        acc = vol_chg - _shift_prev(vol_chg)
        return torch.clamp(acc, -5.0, 5.0)

    @staticmethod
    def pump_deviation(close, window=20):
        pad = torch.zeros((close.shape[0], window - 1), device=close.device)
        c_pad = torch.cat([pad, close], dim=1)
        ma = c_pad.unfold(1, window, 1).mean(dim=-1)
        dev = (close - ma) / (ma + 1e-9)
        return dev

    @staticmethod
    def volatility_clustering(close, window=10):
        """Detect volatility clustering patterns"""
        ret = torch.log(close / (_shift_prev(close) + 1e-9))
        ret_sq = ret ** 2
        
        pad = torch.zeros((ret_sq.shape[0], window - 1), device=close.device)
        ret_sq_pad = torch.cat([pad, ret_sq], dim=1)
        vol_ma = ret_sq_pad.unfold(1, window, 1).mean(dim=-1)
        
        return torch.sqrt(vol_ma + 1e-9)

    @staticmethod
    def momentum_reversal(close, window=5):
        """Capture momentum reversal signals"""
        ret = torch.log(close / (_shift_prev(close) + 1e-9))
        
        pad = torch.zeros((ret.shape[0], window - 1), device=close.device)
        ret_pad = torch.cat([pad, ret], dim=1)
        mom = ret_pad.unfold(1, window, 1).sum(dim=-1)
        
        # Detect reversals
        mom_prev = _shift_prev(mom)
        reversal = (mom * mom_prev < 0).float()
        
        return reversal

    @staticmethod
    def relative_strength(close, high, low, window=14):
        """RSI-like indicator for strength detection"""
        ret = close - _shift_prev(close)
        
        gains = torch.relu(ret)
        losses = torch.relu(-ret)
        
        pad = torch.zeros((gains.shape[0], window - 1), device=close.device)
        gains_pad = torch.cat([pad, gains], dim=1)
        losses_pad = torch.cat([pad, losses], dim=1)
        
        avg_gain = gains_pad.unfold(1, window, 1).mean(dim=-1)
        avg_loss = losses_pad.unfold(1, window, 1).mean(dim=-1)
        
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        return (rsi - 50) / 50  # Normalize


class AdvancedFactorEngineer:
    """Advanced feature engineering with multiple factor types"""
    def __init__(self):
        self.rms_norm = RMSNormFactor(1)
    
    def compute_advanced_features(self, raw_dict, norm_window=120):
        """Compute 12-dimensional feature space with advanced factors"""
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        liq = raw_dict['liquidity']
        fdv = raw_dict['fdv']
        
        # Basic factors — 全部消除未来函数
        ret = torch.log(c / (_shift_prev(c) + 1e-9))
        liq_score = MemeIndicators.liquidity_health(liq, fdv)
        pressure = MemeIndicators.buy_sell_imbalance(c, o, h, l)
        fomo = MemeIndicators.fomo_acceleration(v)
        dev = MemeIndicators.pump_deviation(c)
        log_vol = torch.log1p(v)
        
        # Advanced factors
        vol_cluster = MemeIndicators.volatility_clustering(c)
        momentum_rev = MemeIndicators.momentum_reversal(c)
        rel_strength = MemeIndicators.relative_strength(c, h, l)
        
        # High-low range
        hl_range = (h - l) / (c + 1e-9)
        
        # Close position in range
        close_pos = (c - l) / (h - l + 1e-9)
        
        # Volume trend
        vol_prev = _shift_prev(v)
        vol_trend = (v - vol_prev) / (vol_prev + 1.0)
        
        features = torch.stack([
            rolling_robust_norm(ret, norm_window),
            liq_score,
            pressure,
            rolling_robust_norm(fomo, norm_window),
            rolling_robust_norm(dev, norm_window),
            rolling_robust_norm(log_vol, norm_window),
            rolling_robust_norm(vol_cluster, norm_window),
            momentum_rev,
            rolling_robust_norm(rel_strength, norm_window),
            rolling_robust_norm(hl_range, norm_window),
            close_pos,
            rolling_robust_norm(vol_trend, norm_window)
        ], dim=1)
        
        return features


class FeatureEngineer:
    INPUT_DIM = 6

    @staticmethod
    def compute_features(raw_dict, norm_window=120):
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        liq = raw_dict['liquidity']
        fdv = raw_dict['fdv']
        
        ret = torch.log(c / (_shift_prev(c) + 1e-9))
        liq_score = MemeIndicators.liquidity_health(liq, fdv)
        pressure = MemeIndicators.buy_sell_imbalance(c, o, h, l)
        fomo = MemeIndicators.fomo_acceleration(v)
        dev = MemeIndicators.pump_deviation(c)
        log_vol = torch.log1p(v)

        features = torch.stack([
            rolling_robust_norm(ret, norm_window),
            liq_score,
            pressure,
            rolling_robust_norm(fomo, norm_window),
            rolling_robust_norm(dev, norm_window),
            rolling_robust_norm(log_vol, norm_window)
        ], dim=1)
        
        return features
