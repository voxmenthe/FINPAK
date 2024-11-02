import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Single projection matrix for Q, K, V
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask will be created during forward pass
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(1, 1, 1024, 1024), diagonal=1).bool()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Calculate Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        # Combine heads
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        
        return out

class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Pre-LayerNorm architecture
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.dropout(self.attention(self.ln1(x)))
        # FFN with residual
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

class TimeSeriesDecoder(nn.Module):
    def __init__(
        self,
        d_input: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_outputs: int = 2
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(d_input, d_model)
        
        # Learnable positional embedding instead of sinusoidal
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1024, d_model))
        
        # Stack of decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, n_outputs)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Project input and add positional embeddings
        x = self.input_projection(x)
        x = x + self.pos_embedding[:, :T, :]
        x = self.dropout(x)
        
        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x)
            
        # Use last sequence element for prediction
        x = self.ln_f(x[:, -1])
        return self.output_projection(x)
