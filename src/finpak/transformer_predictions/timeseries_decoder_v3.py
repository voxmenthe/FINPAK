import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Learn relative position embeddings
        self.rel_pos_emb = nn.Parameter(torch.zeros(2 * max_len - 1, d_model))
        
    def forward(self, seq_len: int) -> torch.Tensor:
        # Generate relative position matrix
        positions = torch.arange(seq_len)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel_pos += self.max_len - 1  # Shift to positive indices
        
        # Get embeddings for these positions
        return self.rel_pos_emb[rel_pos]

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_relative_pos: bool = False
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
        
        self.use_relative_pos = use_relative_pos
        if use_relative_pos:
            self.rel_pos = RelativePositionalEncoding(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Calculate Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim) # (B, n_heads, T, T)

        # Add relative positional bias to attention scores
        if self.use_relative_pos:
            rel_pos_bias = self.rel_pos(T)
            # Reshape to match attention heads
            rel_pos_bias = rel_pos_bias.view(T, T, self.n_heads, self.head_dim)
            # Sum over head_dim to get (T, T, n_heads)
            rel_pos_bias = rel_pos_bias.sum(dim=-1)
            rel_pos_bias = rel_pos_bias.permute(2, 0, 1)  # (n_heads, T, T)
            rel_pos_bias = rel_pos_bias.unsqueeze(0) # (1, n_heads, T, T)
            att = att + rel_pos_bias

        # Apply causal mask
        att = att.masked_fill(self.mask[:, :, :T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
            
        # Combine heads
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        
        return out

class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        scales: List[int] = [1, 2, 4],  # Temporal scales
        dropout: float = 0.1,
        bias: bool = True,
        use_relative_pos: bool = False
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scales = scales
        
        # One attention module per scale
        self.attention_modules = nn.ModuleList([
            CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, bias=bias, use_relative_pos=use_relative_pos)
            for _ in scales
        ])
        
        # Output projection to combine multi-scale features
        self.scale_combine = nn.Linear(len(scales) * d_model, d_model, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Process each temporal scale
        scale_outputs = []
        for scale, attention in zip(self.scales, self.attention_modules):
            if scale > 1:
                # Downsample temporally
                x_scaled = F.avg_pool1d(
                    x.transpose(1, 2), 
                    kernel_size=scale, 
                    stride=scale
                ).transpose(1, 2)
                
                # Apply attention
                attended = attention(x_scaled)
                
                # Upsample back to original resolution
                attended = F.interpolate(
                    attended.transpose(1, 2),
                    size=T,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            else:
                attended = attention(x)
                
            scale_outputs.append(attended)
            
        # Combine multi-scale features
        multi_scale = torch.cat(scale_outputs, dim=-1)
        return self.scale_combine(multi_scale)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        scales: List[int] = [1, 2, 4],
        dropout: float = 0.1,
        use_relative_pos: bool = False
    ):
        super().__init__()
        
        # Pre-LayerNorm architecture
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiScaleAttention(
            d_model, 
            n_heads,
            scales=scales,
            dropout=dropout,
            use_relative_pos=use_relative_pos
        )
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
        n_outputs: int = 2,
        use_multi_scale: bool = False,
        use_relative_pos: bool = False,
        temporal_scales: List[int] = [1, 2, 4]
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(d_input, d_model)
        
        # Learnable positional embedding instead of sinusoidal
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1024, d_model))
        
        # Calculate how many unique blocks we need
        # We never reuse first and last blocks, and alternate others
        n_unique_blocks = (n_layers + 3) // 2
        
        # Create the unique decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model, 
                n_heads, 
                d_ff, 
                scales=temporal_scales if use_multi_scale else [1],
                dropout=dropout,
                use_relative_pos=use_relative_pos
            )
            for _ in range(n_unique_blocks)
        ])
        
        # Create a mapping of which block to use at each layer
        self.layer_mapping = self._create_layer_mapping(n_layers)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, n_outputs)
        self.dropout = nn.Dropout(dropout)
        
    def _create_layer_mapping(self, n_layers: int) -> list:
        """Creates a mapping of which block to use at each layer position."""
        # First block is always unique (index 0)
        mapping = [0]
        
        # For the middle blocks, alternate between pairs
        middle_start_idx = 1
        while len(mapping) < n_layers - 1:
            # Add next two indices
            mapping.extend([middle_start_idx, middle_start_idx + 1])
            middle_start_idx += 1
        
        # Trim to size if needed (excluding last position)
        mapping = mapping[:n_layers - 1]
        
        # Last block is always the last unique block
        mapping.append(len(self.blocks) - 1)
        
        return mapping
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Project input and add positional embeddings
        x = self.input_projection(x)
        x = x + self.pos_embedding[:, :T, :]
        x = self.dropout(x)
        
        # Pass through decoder blocks using the layer mapping
        for block_idx in self.layer_mapping:
            x = self.blocks[block_idx](x)
            
        # Use last sequence element for prediction
        x = self.ln_f(x[:, -1])
        return self.output_projection(x)
