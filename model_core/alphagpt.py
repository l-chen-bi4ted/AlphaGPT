import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
from typing import Optional

from .config import ModelConfig, default_config
from .ops import OPS_CONFIG


def _get_vocab_hash() -> str:
    """计算当前 ops_list 的哈希，用于版本校验。"""
    ops_str = ",".join(cfg[0] for cfg in OPS_CONFIG)
    return hashlib.sha256(ops_str.encode()).hexdigest()[:8]


class NewtonSchulzLowRankDecay:
    """LoRD regularization using Newton-Schulz iteration."""

    def __init__(self, named_parameters, decay_rate=1e-3, num_iterations=5, target_keywords=None):
        self.decay_rate = decay_rate
        self.num_iterations = num_iterations
        self.target_keywords = target_keywords or ["qk_norm", "attention"]
        self.params_to_decay = []
        
        for name, param in named_parameters:
            if not param.requires_grad or param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            self.params_to_decay.append((name, param))
    
    @torch.no_grad()
    def step(self):
        for name, W in self.params_to_decay:
            orig_dtype = W.dtype
            X = W.float()
            r, c = X.shape
            
            transposed = False
            if r > c:
                X = X.T
                transposed = True
            
            norm = X.norm() + 1e-8
            X = X / norm
            
            Y = X
            I = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
            
            for _ in range(self.num_iterations):
                A = Y.T @ Y
                Y = 0.5 * Y @ (3.0 * I - A)
            
            if transposed:
                Y = Y.T
            
            W.sub_(self.decay_rate * Y.to(orig_dtype))


class StableRankMonitor:
    def __init__(self, model, target_keywords=None):
        self.model = model
        self.target_keywords = target_keywords or ["q_proj", "k_proj", "attention"]
        self.history = []
    
    @torch.no_grad()
    def compute(self):
        ranks = []
        for name, param in self.model.named_parameters():
            if param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            
            W = param.detach().float()
            S = torch.linalg.svdvals(W)
            stable_rank = (S.norm() ** 2) / (S[0] ** 2 + 1e-9)
            ranks.append(stable_rank.item())
        
        avg_rank = sum(ranks) / len(ranks) if ranks else 0.0
        self.history.append(avg_rank)
        return avg_rank


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class QKNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, 1, 1, d_model) * (d_model ** -0.5))
    
    def forward(self, q, k):
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        return q_norm * self.scale, k_norm * self.scale


class SwiGLU(nn.Module):
    def __init__(self, d_in, d_ff):
        super().__init__()
        self.w = nn.Linear(d_in, d_ff * 2)
        self.fc = nn.Linear(d_ff, d_in)
    
    def forward(self, x):
        x_glu = self.w(x)
        x, gate = x_glu.chunk(2, dim=-1)
        x = x * F.silu(gate)
        return self.fc(x)


class MTPHead(nn.Module):
    def __init__(self, d_model, vocab_size, num_tasks=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(num_tasks)
        ])
        self.task_weights = nn.Parameter(torch.ones(num_tasks) / num_tasks)
        self.task_router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_tasks)
        )
    
    def forward(self, x):
        task_logits = self.task_router(x)
        task_probs = F.softmax(task_logits, dim=-1)
        task_outputs = [head(x) for head in self.task_heads]
        task_outputs = torch.stack(task_outputs, dim=1)
        weighted = (task_probs.unsqueeze(-1) * task_outputs).sum(dim=1)
        return weighted, task_probs


class LoopedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_loops=3, dropout=0.1):
        super().__init__()
        self.num_loops = num_loops
        self.d_model = d_model
        self.nhead = nhead
        
        self.qk_norm = QKNorm(d_model // nhead)
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, is_causal=False):
        for _ in range(self.num_loops):
            x_norm = self.norm1(x)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask, is_causal=is_causal)
            x = x + self.dropout(attn_out)
            
            x_norm = self.norm2(x)
            ffn_out = self.ffn(x_norm)
            x = x + self.dropout(ffn_out)
        
        return x


class LoopedTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_loops=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LoopedTransformerLayer(d_model, nhead, dim_feedforward, num_loops, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None, is_causal=False):
        for layer in self.layers:
            x = layer(x, mask=mask, is_causal=is_causal)
        return x


class AlphaGPT(nn.Module):
    """
    AlphaGPT v3 — 公式生成模型。

    保存/加载时自动记录 vocab_hash，防止 OPS_CONFIG 变更导致 token 语义漂移。
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or default_config
        self.d_model = 64
        self.features_list = ['RET', 'LIQ', 'BUY_SELL', 'FOMO', 'DEV', 'VOL']
        self.ops_list = [cfg[0] for cfg in OPS_CONFIG]
        
        self.vocab = self.features_list + self.ops_list
        self.vocab_size = len(self.vocab)
        self.vocab_hash = _get_vocab_hash()
        
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        max_len = self.config.max_formula_len + 1
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, self.d_model))
        
        self.blocks = LoopedTransformer(
            d_model=self.d_model,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            num_loops=3,
            dropout=0.1
        )
        
        self.ln_f = RMSNorm(self.d_model)
        self.mtp_head = MTPHead(self.d_model, self.vocab_size, num_tasks=3)
        self.head_critic = nn.Linear(self.d_model, 1)

    def forward(self, idx):
        B, T = idx.size()
        
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        
        last_emb = x[:, -1, :]
        logits, task_probs = self.mtp_head(last_emb)
        value = self.head_critic(last_emb)
        
        return logits, value, task_probs

    def save_checkpoint(self, path: str, **extra):
        """保存模型，附带 vocab_hash 用于版本校验。"""
        state = {
            "model": self.state_dict(),
            "vocab": self.vocab,
            "vocab_hash": self.vocab_hash,
            "config": self.config.to_dict(),
        }
        state.update(extra)
        torch.save(state, path)

    @classmethod
    def load_checkpoint(cls, path: str, config: Optional[ModelConfig] = None):
        """加载模型，校验 vocab_hash 一致性。"""
        state = torch.load(path, map_location="cpu")
        current_hash = _get_vocab_hash()
        saved_hash = state.get("vocab_hash", "")
        
        if saved_hash and saved_hash != current_hash:
            raise RuntimeError(
                f"Vocab hash mismatch! Saved={saved_hash} Current={current_hash}. "
                f"OPS_CONFIG may have changed since the model was trained."
            )
        
        model = cls(config=config)
        model.load_state_dict(state["model"])
        return model
