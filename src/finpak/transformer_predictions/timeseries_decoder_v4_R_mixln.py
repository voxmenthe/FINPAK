import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
import math


class HoPE(nn.Module):
  """High-frequency rotary Position Encoding (HoPE) as described in the paper.

  This implementation:
  1. Removes components that would be "activated" within the training length to prevent shortcut learning.
  2. Replaces low-frequency components with position-independent ones.
  3. Retains only high-frequency components for positional encoding.

  Args:
      d_model (int): Dimension of the model (must be divisible by 2).
      base (int): Base for the exponential calculation of frequencies (default: 10000).
  """
  def __init__(self, d_model: int, base: int = 10000):
      super().__init__()
      assert d_model % 2 == 0, "Model dimension must be divisible by 2"
      self.d_model = d_model
      self.base = base

      # Initialize position-independent components for the latter dimensions
      # We'll determine active_comp_id dynamically in forward pass
      self.max_position_independent = d_model // 4  # Reserve up to 1/4 of dims for position-independent
      self.position_independent = nn.Parameter(torch.randn(self.max_position_independent, 2))

  def _calculate_active_components(self, seq_len: int) -> int:
      """Calculate the number of high-frequency components to retain based on sequence length.

      Components where the product of the frequency, sequence length, and 2π is greater than or equal to 1
      are considered 'active' and are retained.

      Args:
          seq_len (int): Length of the input sequence.

      Returns:
          int: Number of active components (high-frequency components).
      """
      dim = self.d_model // 2
      positions = torch.arange(dim, dtype=torch.float32, device=self.position_independent.device)
      freqs = 1.0 / (self.base ** (positions / dim))
      # Calculate the condition θ_i * L ≥ 1, where θ_i = freq_i * 2π
      condition = (freqs * 2 * math.pi * seq_len) >= 1
      active_comp = int(condition.sum().item())
      # Ensure we don't exceed max_position_independent for the remaining dims
      return min(active_comp, dim - self.max_position_independent)

  def _get_freq_bands(self, active_comp_id: int) -> torch.Tensor:
      """Generate frequency bands for the high-frequency components only.

      Args:
          active_comp_id (int): Number of active components to use.

      Returns:
          torch.Tensor: Frequencies for high-frequency components.
      """
      dim = self.d_model // 2
      positions = torch.arange(dim, dtype=torch.float32, device=self.position_independent.device)
      freqs = 1.0 / (self.base ** (positions / dim))
      # Retain only the active components
      freqs = freqs[:active_comp_id]
      return freqs.unsqueeze(0)  # Shape: [1, active_comp_id]

  def forward(self, seq_len: int) -> torch.Tensor:
      """Generate relative positional encodings for sequences of the given length.

      Args:
          seq_len (int): Length of the input sequence.

      Returns:
          torch.Tensor: Relative positional encodings of shape [seq_len, seq_len, d_model].
      """
      # Dynamically calculate active components based on sequence length
      active_comp_id = self._calculate_active_components(seq_len)
      freq_bands = self._get_freq_bands(active_comp_id)
      device = self.position_independent.device

      # Create relative positions matrix
      positions = torch.arange(seq_len, device=device)
      rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # Shape: [seq_len, seq_len]

      # Calculate the angles (theta) for high-frequency components
      rel_pos_expanded = rel_pos.unsqueeze(-1)  # Shape: [seq_len, seq_len, 1]
      theta = rel_pos_expanded * freq_bands  # Shape: [seq_len, seq_len, active_comp_id]

      # Apply rotary encoding (cosine and sine)
      cos_theta = torch.cos(theta)  # Shape: [seq_len, seq_len, active_comp_id]
      sin_theta = torch.sin(theta)  # Shape: [seq_len, seq_len, active_comp_id]

      # Stack cos and sin components
      high_freq = torch.stack([cos_theta, sin_theta], dim=-1)  # Shape: [seq_len, seq_len, active_comp_id, 2]

      # Calculate how many position-independent components we need
      latter_dim = (self.d_model // 2) - active_comp_id
      pos_independent = self.position_independent[:latter_dim]

      # Expand position-independent components to match sequence dimensions
      pos_independent = pos_independent.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, latter_dim, 2]
      pos_independent = pos_independent.expand(seq_len, seq_len, -1, -1)  # Shape: [seq_len, seq_len, latter_dim, 2]

      # Concatenate high-frequency and position-independent components
      combined = torch.cat([high_freq, pos_independent], dim=2)  # Shape: [seq_len, seq_len, d_model//2, 2]

      # Reshape to final dimension
      return combined.reshape(seq_len, seq_len, self.d_model)


class MixLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.post_ln = nn.LayerNorm(normalized_shape, eps=eps)
        self.pre_ln = nn.LayerNorm(normalized_shape, eps=eps)
    
    def forward(self, x: torch.Tensor, is_pre_ln: bool) -> torch.Tensor:
        if is_pre_ln:
            return self.pre_ln(x)
        else:
            return self.post_ln(x)


class CausalSelfAttention(nn.Module):
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

      # Single projection matrix for Q, K, V
      self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
      self.proj = nn.Linear(d_model, d_model, bias=bias)
      self.dropout = nn.Dropout(dropout)

      # Causal mask will be created during forward pass
      self.register_buffer(
          "mask",
          torch.triu(torch.ones(1, 1, 1024, 1024), diagonal=1).bool()
      )

      # Positional encoding settings
      self.pos_enc = HoPE(d_model, base=base)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      B, T, C = x.shape

      # Calculate Q, K, V
      qkv = self.qkv(x)
      qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
      qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
      q, k, v = qkv[0], qkv[1], qkv[2]

      # Scaled dot-product attention with causal mask
      att = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim) # (B, n_heads, T, T)

      # Add positional bias to attention scores if using positional encoding
      pos_bias = self.pos_enc(T)
      # Reshape to match attention heads
      pos_bias = pos_bias.view(T, T, self.n_heads, self.head_dim)
      # Sum over head_dim to get (T, T, n_heads)
      pos_bias = pos_bias.sum(dim=-1)
      pos_bias = pos_bias.permute(2, 0, 1)  # (n_heads, T, T)
      pos_bias = pos_bias.unsqueeze(0) # (1, n_heads, T, T)
      att = att + pos_bias

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
        base: int = 10000
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scales = scales
        
        # One attention module per scale
        self.attention_modules = nn.ModuleList([
            CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, bias=bias, base=base)
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
        mixln_alpha: float = 0.25, # Mix-LN hyperparameter
        base: int = 10000
    ):
        super().__init__()
        self.d_model = d_model
        self.mixln_alpha = mixln_alpha
        
        # Mix-LN replaces LayerNorm
        self.ln1 = MixLayerNorm(d_model)
        self.attention = MultiScaleAttention(
            d_model, 
            n_heads,
            scales=scales,
            dropout=dropout,
            base=base
        )
        self.ln2 = MixLayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, layer_idx: int, total_layers: int) -> torch.Tensor:
        # Determine if current layer is Pre-LN or Post-LN based on alpha
        if layer_idx < int(self.mixln_alpha * total_layers):
            # Post-LN
            x_norm = self.ln1.forward(x + self.dropout(self.attention(x)), is_pre_ln=False)
            x = x_norm + self.dropout(self.mlp(x_norm)) # Residual
            return x
        else:
            # Pre-LN
            x_norm = self.ln1.forward(x, is_pre_ln=True)
            x = x + self.dropout(self.attention(x_norm))
            x_norm = self.ln2.forward(x, is_pre_ln=True)
            x = x + self.dropout(self.mlp(x_norm)) # Residual
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
        temporal_scales: List[int] = [1, 2, 4],
        mixln_alpha: float = 0.25, # Mix-LN hyperparameter
        base: int = 10000
    ):
        super().__init__()

        self.d_model = d_model
        self.input_projection = nn.Linear(d_input, d_model)
        self.n_layers = n_layers # Store the number of layers

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
                mixln_alpha=mixln_alpha,
                base=base
            )
            for _ in range(n_unique_blocks)
        ])
        
        # Create a mapping of which block to use at each layer
        self.layer_mapping = self._create_layer_mapping(n_layers)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, n_outputs)
        self.dropout = nn.Dropout(dropout)
        
    def _create_layer_mapping(self, n_layers: int) -> list:
        """Creates a mapping of which block to use at each layer position.
        
        The first and last layers always use unique blocks. For middle layers:
        - If number of middle layers is divisible by 2, uses alternating pairs
        - If number of middle layers is divisible by 3, uses alternating triples
        - Otherwise uses a combination of pairs and triples, with unique blocks for remainder
        
        Examples:
            n_layers=4: [0, 1, 2, 3]           # All unique
            n_layers=6: [0, 1, 2, 1, 2, 3]     # Middle uses pairs
            n_layers=8: [0, 1, 2, 3, 1, 2, 3, 4] # Middle uses triples
            n_layers=7: [0, 1, 2, 1, 2, 3, 4]  # Middle uses pairs + unique
        """
        # First and last blocks are always unique
        mapping = [0]
        next_unique_idx = 1
        
        # Calculate middle layers
        middle_layers = n_layers - 2
        
        if middle_layers <= 0:
            mapping.append(next_unique_idx)
            return mapping
        
        # Try to fill with triples first if divisible by 3
        if middle_layers % 3 == 0:
            triple_sets = middle_layers // 3
            triple_blocks = [next_unique_idx, next_unique_idx + 1, next_unique_idx + 2]
            for _ in range(triple_sets):
                mapping.extend(triple_blocks)
            next_unique_idx += 3
            
        # Try to fill with pairs if divisible by 2
        elif middle_layers % 2 == 0:
            pair_sets = middle_layers // 2
            pair_blocks = [next_unique_idx, next_unique_idx + 1]
            for _ in range(pair_sets):
                mapping.extend(pair_blocks)
            next_unique_idx += 2
            
        # Handle mixed case
        else:
            # Try to fit as many triples as possible
            triples_to_use = middle_layers // 3
            remaining = middle_layers % 3
            
            if triples_to_use > 0:
                triple_blocks = [next_unique_idx, next_unique_idx + 1, next_unique_idx + 2]
                for _ in range(triples_to_use):
                    mapping.extend(triple_blocks)
                next_unique_idx += 3
            
            # Fill remaining with pairs if possible
            if remaining >= 2:
                pair_blocks = [next_unique_idx, next_unique_idx + 1]
                mapping.extend(pair_blocks)
                next_unique_idx += 2
                remaining -= 2
            
            # Any remaining positions get unique blocks
            while remaining > 0:
                mapping.append(next_unique_idx)
                next_unique_idx += 1
                remaining -= 1
        
        # Add final unique block
        mapping.append(next_unique_idx)
        
        return mapping
        
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape  # batch size, sequence length, channels
        
        # Project input to model dimension
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Pass through decoder blocks using the layer mapping with residual connections
        original_x = x
        for i, block_idx in enumerate(self.layer_mapping):
            x = self.blocks[block_idx].forward(x, i, self.n_layers) + original_x
            original_x = x
        
        # Use last sequence element for prediction
        x = self.ln_f(x[:, -1])
        return self.output_projection(x)
