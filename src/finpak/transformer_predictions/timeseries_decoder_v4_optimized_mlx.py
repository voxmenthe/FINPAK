import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

import mlx.core as mx
import mlx.nn as mlx_nn


class HoPEMLX(mlx_nn.Module):
    """High-frequency rotary Position Encoding (HoPE) implemented in MLX.
    
    This implementation:
    1. Removes components that would be "activated" within the training length
    2. Replaces low-frequency components with position-independent ones
    3. Retains only high-frequency components for positional encoding
    
    Args:
        d_model (int): Dimension of the model (must be divisible by 2)
        base (int): Base for exponential calculation of frequencies (default: 10000)
    """
    def __init__(
        self,
        d_model: int,
        max_position_independent: int = 0,
        base: float = 10000.0,
        theta_type: str = "log",
        cumsum_pow: bool = True,
    ):
        super().__init__()
        assert d_model % 2 == 0, "Model dimension must be divisible by 2"
        self.max_position_independent = max_position_independent
        self.d_model = d_model
        self.base = base
        self.theta_type = theta_type
        self.cumsum_pow = cumsum_pow

        # Initialize position-independent components [d_model//2, 2]
        self.position_independent = mx.random.normal(
            (self.d_model // 2, 2),  # Initialize with maximum possible size
            scale=0.02
        )

        # Initialize theta based on type
        if theta_type == "log":
            self.theta = 1.0 / (base ** (mx.arange(d_model) / d_model))
        else:  # "linear"
            self.theta = 1.0 / mx.linspace(1.0, base, d_model)
    
    def _calculate_active_components(self, active_comp: int) -> int:
        """Calculate the number of active components."""
        result = mx.minimum(active_comp, self.d_model - self.max_position_independent)
        # Convert to Python int
        return int(result.item())
    
    def _get_freq_bands(self, active_comp_id: int) -> mx.array:
        """Generate frequency bands for high-frequency components only."""
        dim = self.d_model // 2
        positions = mx.arange(dim, dtype=mx.float32)  # [dim]
        freqs = 1.0 / (self.base ** (positions / dim))  # [dim]
        # Convert active_comp_id to Python int if it's an MLX array
        if isinstance(active_comp_id, mx.array):
            active_comp_id = int(active_comp_id.item())
        freqs = freqs[:active_comp_id]  # [active_comp_id]
        # Add necessary dimensions for broadcasting
        return mx.expand_dims(mx.expand_dims(freqs, 0), 0)  # [1, 1, active_comp_id]
    
    def __call__(self, seq_len: int) -> mx.array:
        """Generate relative positional encodings for sequences of given length.
        
        Args:
            seq_len (int): Length of input sequence
            
        Returns:
            mx.array: Positional encodings of shape [seq_len, seq_len, d_model]
        """
        # Get active high-frequency components
        active_comp_id = self._calculate_active_components(seq_len)
        freq_bands = self._get_freq_bands(active_comp_id)  # [1, 1, active_comp_id]
        
        # Create relative positions matrix [seq_len, seq_len]
        positions = mx.arange(seq_len)  # [seq_len]
        rel_pos = mx.expand_dims(positions, 1) - mx.expand_dims(positions, 0)  # [seq_len, seq_len]
        
        # Calculate angles (theta) for high-frequency components
        rel_pos_expanded = mx.expand_dims(rel_pos, -1)  # [seq_len, seq_len, 1]
        theta = rel_pos_expanded * freq_bands  # [seq_len, seq_len, active_comp_id]
        
        # Apply rotary encoding (cosine and sine)
        cos_theta = mx.cos(theta)  # [seq_len, seq_len, active_comp_id]
        sin_theta = mx.sin(theta)  # [seq_len, seq_len, active_comp_id]
        
        # Stack cos and sin components [seq_len, seq_len, active_comp_id, 2]
        high_freq = mx.stack([cos_theta, sin_theta], axis=-1)
        
        # Calculate remaining dimensions for position-independent components
        latter_dim = (self.d_model // 2) - active_comp_id
        
        if latter_dim > 0:
            # Get position-independent components [latter_dim, 2]
            # Always slice from the end to maintain consistency
            pos_ind = self.position_independent[-latter_dim:]
            
            # Debug print to check dimensions
            # print(f"latter_dim: {latter_dim}")
            # print(f"pos_ind shape: {pos_ind.shape}")
            # print(f"position_independent shape: {self.position_independent.shape}")
            
            # Create empty tensor for position-independent components
            pos_independent = mx.zeros((seq_len, seq_len, latter_dim, 2))
            
            # Fill the position-independent components
            # Broadcasting: [latter_dim, 2] -> [1, 1, latter_dim, 2] -> [seq_len, seq_len, latter_dim, 2]
            pos_independent = pos_independent + mx.expand_dims(mx.expand_dims(pos_ind, 0), 0)
            
            # Concatenate along the frequency dimension
            combined = mx.concatenate([high_freq, pos_independent], axis=2)
        else:
            combined = high_freq
        
        # Reshape to final dimension [seq_len, seq_len, d_model]
        return mx.reshape(combined, (seq_len, seq_len, self.d_model))


class CausalSelfAttentionMLX(mlx_nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        base: int = 10000
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / mx.sqrt(self.head_dim)  # Pre-compute scale
        
        # Single projection matrix for Q, K, V like in Qwen2
        self.qkv = mlx_nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = mlx_nn.Linear(d_model, d_model, bias=False)  # No bias needed for output proj
        
        # Use MLX's built-in RoPE instead of custom HoPE
        self.rope = mlx_nn.RoPE(
            self.head_dim,
            traditional=False,
            base=base,
        )
        
        self.dropout = mlx_nn.Dropout(dropout)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, D = x.shape

        # Compute Q, K, V with single projection like Qwen2
        qkv = self.qkv(x)
        qkv = mx.reshape(qkv, (B, L, 3, self.n_heads, self.head_dim))
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        # Scaled dot-product attention
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale

        # Create and apply causal mask more efficiently
        if mask is None:
            mask = mx.triu(mx.ones((L, L)), k=1)
            scores = mx.where(mask, float('-inf'), scores)

        # Apply softmax with better precision
        scores = mx.softmax(scores, axis=-1, precise=True)
        scores = self.dropout(scores)

        # Compute attention output
        out = scores @ v
        out = mx.transpose(out, (0, 2, 1, 3))
        out = mx.reshape(out, (B, L, D))
        
        return self.proj(out)


class MultiScaleAttentionMLX(mlx_nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        scales: List[int] = [1, 2, 4],
        dropout: float = 0.1,
        bias: bool = True,
        base: int = 10000
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scales = scales
        
        # Store attention modules in a regular list
        # MLX will track parameters automatically through the parent module
        self.attention_modules = [
            CausalSelfAttentionMLX(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                bias=bias,
                base=base
            )
            for _ in scales
        ]
        
        self.scale_combine = mlx_nn.Linear(len(scales) * d_model, d_model, bias=bias)
        
    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        
        # Process each temporal scale
        scale_outputs = []
        for scale, attention in zip(self.scales, self.attention_modules):
            if scale > 1:
                # Downsample temporally using average pooling
                x_t = mx.transpose(x, (0, 2, 1))
                downsampled_len = T // scale
                x_scaled = mx.mean(
                    mx.reshape(x_t, (B, C, downsampled_len, scale)), 
                    axis=-1
                )
                x_scaled = mx.transpose(x_scaled, (0, 2, 1))
                
                # Apply attention
                attended = attention(x_scaled)
                
                # Upsample back to original resolution
                attended_t = mx.transpose(attended, (0, 2, 1))  # [B, C, downsampled_len]
                
                # Linear interpolation
                indices = mx.arange(T, dtype=mx.float32) * (downsampled_len - 1) / (T - 1)
                lower_idx = mx.floor(indices).astype(mx.int32)
                upper_idx = mx.minimum(lower_idx + 1, downsampled_len - 1)
                alpha = indices - lower_idx
                
                # Gather values and interpolate
                lower_vals = attended_t[:, :, lower_idx]
                upper_vals = attended_t[:, :, upper_idx]
                alpha = mx.reshape(alpha, (1, 1, -1))
                
                # Linear interpolation
                interpolated = lower_vals * (1 - alpha) + upper_vals * alpha
                attended = mx.transpose(interpolated, (0, 2, 1))
            else:
                attended = attention(x)
                
            scale_outputs.append(attended)
            
        # Combine multi-scale features
        multi_scale = mx.concatenate(scale_outputs, axis=-1)
        return self.scale_combine(multi_scale)


class DecoderBlockMLX(mlx_nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        base: int = 10000
    ):
        super().__init__()
        
        # Use RMSNorm like in Gemma2 for better performance
        self.ln1 = mlx_nn.RMSNorm(d_model)
        self.attention = CausalSelfAttentionMLX(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            base=base
        )
        self.ln2 = mlx_nn.RMSNorm(d_model)
        
        # MLP block using Gemma2's approach
        self.mlp = mlx_nn.Sequential(
            mlx_nn.Linear(d_model, d_ff, bias=False),
            mlx_nn.GELU(),  # Using GELU like in modern transformers
            mlx_nn.Linear(d_ff, d_model, bias=False)
        )
        
        self.dropout = mlx_nn.Dropout(dropout)
        
    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Pre-LayerNorm architecture with residual connections
        h = self.ln1(x)
        h = self.attention(h, mask)
        h = self.dropout(h)
        x = x + h
        
        h = self.ln2(x)
        h = self.mlp(h)
        h = self.dropout(h)
        x = x + h
        
        return x


class TimeseriesDecoderMLX(mlx_nn.Module):
    def __init__(
        self,
        d_input: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_outputs: int = 2,
        base: int = 10000
    ):
        super().__init__()
        
        self.input_projection = mlx_nn.Linear(d_input, d_model)
        
        # Create decoder blocks
        self.blocks = [
            DecoderBlockMLX(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                base=base
            )
            for _ in range(n_layers)
        ]
        
        # Final normalization and projection
        self.ln_f = mlx_nn.RMSNorm(d_model)
        self.output_projection = mlx_nn.Linear(d_model, n_outputs, bias=False)
        
    def __call__(self, x: mx.array) -> mx.array:
        # Project input
        x = self.input_projection(x)
        
        # Process through decoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Final prediction using last token
        x = self.ln_f(x[:, -1])
        return self.output_projection(x)