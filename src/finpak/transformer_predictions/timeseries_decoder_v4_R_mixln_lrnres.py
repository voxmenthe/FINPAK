import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
import math

class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class HoPE(nn.Module):
  """High-frequency rotary Position Encoding (HoPE) as described in the paper.

  Dynamically determines how many high-frequency componentsto keep based on the sequence length, then uses random param 'position_independent'for the remainder (low-frequency) dimensions.

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
      # Up to 1/4 of dims can be purely position-independent
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
      # Keep only those freq_i where freq_i * 2π * seq_len >= 1
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

      # Position-independent remainder
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
    """
    Mix of pre-LayerNorm and post-LayerNorm, based on a flag.
    Can possiblyexpand this to add RMS Norm.
    """
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
      base: int = 10000,
      max_seq_len: int = 256
  ):
      super().__init__()
      assert d_model % n_heads == 0

      self.d_model = d_model
      self.n_heads = n_heads
      self.head_dim = d_model // n_heads

      # Separate linear layers for Q, K, V
      self.c_q = nn.Linear(d_model, d_model, bias=True)
      self.c_k = nn.Linear(d_model, d_model, bias=True)
      self.c_v = nn.Linear(d_model, d_model, bias=True)
      self.proj = nn.Linear(d_model, d_model, bias=True)

      # Learned gating parameters for ‘v’ only
      self.lamb1 = nn.Parameter(torch.tensor(0.5))
      self.lamb2 = nn.Parameter(torch.tensor(0.5))

      self.dropout = nn.Dropout(dropout)

      # For causal masking # need to update to pass through max_seq_len all the way
      self.register_buffer(
          "mask",
          torch.triu(torch.ones(1, 1, max_seq_len, max_seq_len), diagonal=1).bool(),
          persistent=False
      )

      # Positional encoding settings
      self.pos_enc = HoPE(d_model, base=base)

  def forward(self, x: torch.Tensor, v1: torch.Tensor=None) -> torch.Tensor:
      """
      x:  shape [B, T, d_model]
      v1: shape [B, T, n_heads, head_dim] from a previous iteration (optional).
              If None, we default to new v.  The final returned 'new_v' can be fed 
              into the next block if you want iterative gating across layers.
      
      Returns:
          out:  shape [B, T, d_model]
          new_v: shape [B, T, n_heads, head_dim]
      """
      B, T, _ = x.shape

      # Project for Q, K, V
      q = self.c_q(x).view(B, T, self.n_heads, self.head_dim)
      k = self.c_k(x).view(B, T, self.n_heads, self.head_dim)
      v = self.c_v(x).view(B, T, self.n_heads, self.head_dim)
  
      # If v1 is not provided, use newly computed v
      if v1 is None:
          v1 = v
      # Weighted combination of old v1 and new v
      v = self.lamb1 * v + self.lamb2 * v1.view_as(v)
  
      # Compute attention scores
      scale = 1.0 / (self.head_dim ** 0.5)
      att_scores = torch.einsum('bthd,bThd->bhtT', q, k) * scale  # shape [B, n_heads, T, T]
  
      # Add HoPE-based positional bias
      pos_bias = self.pos_enc(T)  # shape [T, T, d_model]
      pos_bias = pos_bias.view(T, T, self.n_heads, self.head_dim)  # [T, T, n_heads, head_dim]
      # sum over last dim => shape [T, T, n_heads], then rearrange to [n_heads, T, T]
      pos_bias = pos_bias.sum(dim=-1).permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, T, T]
      att_scores = att_scores + pos_bias
  
      # Apply causal mask
      att_scores = att_scores.masked_fill(self.mask[:, :, :T, :T], float('-inf'))
  
      # Softmax
      att_weights = F.softmax(att_scores, dim=-1)
      att_weights = self.dropout(att_weights)
  
      # Weighted sum for the final
      out = torch.einsum('bhtT,bThd->bthd', att_weights, v)  # [B, T, n_heads, head_dim]
      out = out.reshape(B, T, self.d_model)
      out = self.proj(out)
  
      return out, v


class MultiScaleAttention(nn.Module):
    """
    If you want multi-scale attention, we must decide how “v1” gating applies across multiple scales. Below, we show a simplistic approach where we reuse the same v1 for each scale and return only the last scale’s v as new_v. You can refine if you want to track a separate “v1” per scale.
    """
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
        
    def forward(self, x: torch.Tensor, v1: torch.Tensor=None) -> torch.Tensor:
        """
        x: Shape [B, T, d_model]
        v1: shape [B, T, n_heads, head_dim], if you're carrying it across blocks

        Returns:
          combined_out: [B, T, d_model]
          new_v:        [B, T, n_heads, head_dim]
        """
        B, T, C = x.shape
        new_v_global = None
        
        # Process each temporal scale
        scale_outputs = []
        for scale_idx, (scale, attn) in enumerate(zip(self.scales, self.attention_modules)):
            # Downsample if scale > 1
            if scale > 1:
                x_scaled = F.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=scale,
                    stride=scale
                ).transpose(1, 2)
            else:
                x_scaled = x

            # For each scale, do attention
            attn_out, new_v = attn(x_scaled, v1)
            # We simply overwrite v1 with new_v from the last operation (one approach)
            v1 = new_v

            # Upsample back if scale > 1
            if scale > 1:
                attn_out = F.interpolate(
                    attn_out.transpose(1, 2),
                    size=T,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            scale_outputs.append(attn_out)
            
        # We keep the last new_v as the final for the next block
        new_v_global = v1

        multi_scale = torch.cat(scale_outputs, dim=-1)  # [B, T, len(scales)*d_model]
        combined_out = self.scale_combine(multi_scale)  # [B, T, d_model]
        return combined_out, new_v_global


class DecoderBlock(nn.Module):
    """
    DecoderBlock: MixLayerNorm + ReLU² MLP + v1 Pipeline
    We integrate the gating approach by receiving and returning “v1.” 
    The “v1” can be threaded from layer to layer at your discretion.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        scales: List[int] = [1, 2, 4],
        dropout: float = 0.1,
        mixln_alpha: float = 0.25, # fraction of layers that do post-LN vs. pre-LN
        base: int = 10000
    ):
        super().__init__()
        self.d_model = d_model
        self.mixln_alpha = mixln_alpha
        
        # Mix-LN replaces LayerNorm
        self.ln1 = MixLayerNorm(d_model)
        # Multi-scale or single-scale attention
        self.attention = MultiScaleAttention(d_model, n_heads, scales, dropout, base)
        self.ln2 = MixLayerNorm(d_model)

        # GPT-like MLP with ReLU²
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout),
            LambdaLayer(lambda t: F.relu(t).square()),  # ReLU²
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, layer_idx: int, total_layers: int, v1: torch.Tensor=None):
        """
        x: shape [B, T, d_model]
        v1: shape [B, T, n_heads, head_dim], possibly from a previous iteration or block
        """
        # Determine if this is post-LN or pre-LN
        is_post = (layer_idx < int(self.mixln_alpha * total_layers))

        # Post-LN path
        if is_post:
            # 1) x + attention => LN => MLP
            attn_out, new_v = self.attention(x, v1)
            x_att = x + attn_out
            x_norm = self.ln1(x_att, is_pre_ln=False)
            mlp_out = x_norm + self.mlp(x_norm)
            return mlp_out, new_v

        # Pre-LN path
        else:
            # LN => attention => residual => LN => MLP => residual
            x_norm = self.ln1(x, is_pre_ln=True)
            attn_out, new_v = self.attention(x_norm, v1)
            x_att = x + attn_out  # residual #1

            x_att_norm = self.ln2(x_att, is_pre_ln=True)
            mlp_out = x_att + self.mlp(x_att_norm)  # residual #2
            return mlp_out, new_v


class TimeSeriesDecoder(nn.Module):
    """
    This is the main decoder that arranges blocks in the order you specified, plus our layer-mapping approach (or a simpler “one block per layer” approach). Each call to a block returns (x, new_v), which we feed forward to the next layer.
    """
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
          - The first and last blocks are unique.
          - Middle layers may be repeated in pairs or triples
        
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
        """
        x: shape [B, T, d_input]
        Returns [B, n_outputs]
        """
        B, T, _ = x.shape  # batch size, sequence length, channels
        
        # Project input to model dimension
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Pass through decoder blocks using the layer mapping with residual connections, passing v1 forward
        v1 = None
        for i, block_idx in enumerate(self.layer_mapping):
            x, new_v = self.blocks[block_idx](x, i, self.n_layers, v1)
            v1 = new_v  # carry v forward for gating in the next layer
        
        # Use last sequence element for prediction
        x = self.ln_f(x[:, -1])
        return self.output_projection(x)
