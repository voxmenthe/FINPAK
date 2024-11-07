import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

class Rotary(nn.Module):
    """Rotary positional embeddings for improved attention."""
    
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        
    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to input tensor."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1).type_as(x)

class ModernCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        
        # Separate Q,K,V projections without bias
        self.c_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.c_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.c_v = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Output projection with zero init
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.c_proj.weight.data.zero_()
        
        # Rotary embeddings
        self.rotary = Rotary(self.head_dim)
        
        # Learnable scaling parameter
        self.lamb = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x, v1=None):
        B, T, C = x.size()
        
        # Project to Q, K, V
        q = self.c_q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_heads, self.head_dim)
        
        # Handle first block case
        if v1 is None:
            v1 = v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)
        
        # Apply rotary embeddings
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        
        # Efficient attention using PyTorch's built-in function
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), 
            k.transpose(1, 2), 
            v.transpose(1, 2),
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y, v1

class ModernMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=False)
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=False)
        self.c_proj.weight.data.zero_()
        
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU²
        x = self.c_proj(x)
        return x

class ModernDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = ModernCausalSelfAttention(config)
        self.mlp = ModernMLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        
    def forward(self, x, v1, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(F.rms_norm(x, (x.size(-1),)), v1)
        x = x + x1
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x, v1

@dataclass
class TimeSeriesConfig:
    d_input: int = 3
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    n_outputs: int = 2

class ModernTimeSeriesDecoder(nn.Module):
    """Drop-in replacement for TimeSeriesDecoder with modern architecture improvements."""
    
    def __init__(
        self,
        d_input: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,  # Ignored but kept for compatibility
        dropout: float = 0.1,
        n_outputs: int = 2
    ):
        super().__init__()
        
        # Create config object
        self.config = TimeSeriesConfig(
            d_input=d_input,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            n_outputs=n_outputs
        )
        
        # Input projection
        self.input_projection = nn.Linear(d_input, d_model, bias=False)
        
        # Stack of modern decoder blocks
        self.blocks = nn.ModuleList([
            ModernDecoderBlock(self.config) for _ in range(n_layers)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, n_outputs, bias=False)
        self.output_projection.weight.data.zero_()
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Project input
        x = self.input_projection(x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        v1 = None
        
        # Pass through decoder blocks
        for block in self.blocks:
            x, v1 = block(x, v1, x0)
            
        # Final layernorm and prediction
        x = F.rms_norm(x, (x.size(-1),))
        x = x[:, -1]  # Take last position only
        x = self.output_projection(x)
        x = 30 * torch.tanh(x / 30)  # Bounded output range
        
        return x
