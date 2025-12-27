import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Tuple


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


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Create a small cache of relative position embeddings that we'll expand as needed
        initial_size = 32  # Start with a small size
        self.max_cached_len = initial_size
        self.rel_pos_emb = nn.Parameter(torch.zeros(2 * initial_size - 1, d_model))
        
    def _expand_cache(self, new_max_len: int):
        """Expand the cached relative position embeddings to accommodate longer sequences."""
        old_max_len = self.max_cached_len
        device = self.rel_pos_emb.device
        
        # Create new larger embedding
        new_emb = nn.Parameter(torch.zeros(2 * new_max_len - 1, self.d_model, device=device))
        
        # Copy old values to center of new embedding
        old_start = new_max_len - old_max_len
        new_emb[old_start:old_start + len(self.rel_pos_emb)] = self.rel_pos_emb
        
        # Initialize new positions with interpolated values
        if old_max_len < new_max_len:
            # Initialize new positions before old values
            if old_start > 0:
                new_emb[:old_start] = self.rel_pos_emb[0].unsqueeze(0) + \
                    (self.rel_pos_emb[0] - self.rel_pos_emb[1]).unsqueeze(0) * \
                    torch.arange(old_start, 0, -1, device=device).unsqueeze(1)
            
            # Initialize new positions after old values
            remaining = len(new_emb) - (old_start + len(self.rel_pos_emb))
            if remaining > 0:
                new_emb[-remaining:] = self.rel_pos_emb[-1].unsqueeze(0) + \
                    (self.rel_pos_emb[-1] - self.rel_pos_emb[-2]).unsqueeze(0) * \
                    torch.arange(1, remaining + 1, device=device).unsqueeze(1)
        
        self.rel_pos_emb = new_emb
        self.max_cached_len = new_max_len
        
    def forward(self, seq_len: int) -> torch.Tensor:
        # Expand cache if needed
        if seq_len > self.max_cached_len:
            self._expand_cache(seq_len)
        
        # Generate relative position matrix
        positions = torch.arange(seq_len, device=self.rel_pos_emb.device)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel_pos += self.max_cached_len - 1  # Shift to positive indices
        
        # Get embeddings for these positions
        return self.rel_pos_emb[rel_pos]


class CausalSelfAttention(nn.Module):
  def __init__(
      self,
      d_model: int,
      n_heads: int,
      dropout: float = 0.1,
      bias: bool = True,
      use_relative_pos: bool = False,
      use_hope_pos: bool = False,
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
      assert not (use_relative_pos and use_hope_pos), "Cannot use both RelativePositionalEncoding and HoPE"
      self.use_pos_encoding = use_relative_pos or use_hope_pos

      if use_relative_pos:
          self.pos_enc = RelativePositionalEncoding(d_model)
      elif use_hope_pos:
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
      if self.use_pos_encoding:
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
        use_relative_pos: bool = False,
        use_hope_pos: bool = False,
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
            CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, bias=bias, use_relative_pos=use_relative_pos, use_hope_pos=use_hope_pos, base=base)
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
        use_relative_pos: bool = False,
        use_hope_pos: bool = False,
        base: int = 10000
    ):
        super().__init__()
        
        # Pre-LayerNorm architecture
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiScaleAttention(
            d_model, 
            n_heads,
            scales=scales,
            dropout=dropout,
            use_relative_pos=use_relative_pos,
            use_hope_pos=use_hope_pos,
            base=base
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

class HybridInputEmbedding(nn.Module):
    """
    Combines continuous and categorical features into a single embedding
    
    Args:
        d_continuous: Number of continuous features
        n_categorical: Number of categorical features
        n_bins: Number of bins for each categorical feature
        d_model: Model dimension
        dropout: Dropout rate
    """
    def __init__(self, d_continuous: int, n_categorical: int, n_bins: int, d_model: int, dropout: float = 0.1, debug: bool = False):
        """
        Initialize the hybrid embedding layer that combines continuous and categorical features.
        
        Design decisions:
        1. Dynamic embedding dimension:
           - Categorical embedding size scales with d_model
           - Each categorical feature gets a proportional share of d_model
           - Remaining space allocated to continuous features
        
        2. Space allocation example (for d_model=64):
           - Total space = 64 dimensions
           - If n_categorical = 1:
             * Categorical space = min(16, d_model//4) = 16 dims
             * Continuous space = 48 dims (remaining space)
        
        3. Feature handling:
           - Continuous features: project from d_continuous to d_continuous_proj
           - Categorical features: each gets cat_embed_dim dimensions
        
        Args:
            d_continuous (int): Number of continuous input features (e.g., 21)
            n_categorical (int): Number of categorical features (e.g., 1)
            n_bins (int): Number of bins for each categorical feature (e.g., 10)
            d_model (int): Target model dimension (must be even)
            dropout (float): Dropout rate
        """
        super().__init__()
        self.debug = debug
        assert d_model % 2 == 0, "d_model must be even"
        
        # Calculate embedding dimensions dynamically based on d_model
        if n_categorical > 0:
            # Maximum space for all categorical features combined (25% of d_model)
            max_categorical_space = d_model // 4
            
            # Calculate embedding dim per categorical feature:
            # 1. At most d_model//4 per feature
            # 2. At most max_categorical_space divided by number of features
            # 3. At least 4 dimensions
            self.cat_embed_dim = max(4, min(d_model // 4, max_categorical_space // n_categorical))
            
            # Total space for all categorical features
            self.d_categorical_proj = n_categorical * self.cat_embed_dim
            
            # Remaining space for continuous features
            self.d_continuous_proj = d_model - self.d_categorical_proj
            
            assert self.d_continuous_proj > 0, \
                f"d_model ({d_model}) too small for {n_categorical} categorical features. " \
                f"Needed {self.d_categorical_proj} dimensions for categorical, only {self.d_continuous_proj} left for continuous."
        else:
            self.cat_embed_dim = 0
            self.d_continuous_proj = d_model
            self.d_categorical_proj = 0
        
        if self.debug:
            print(f"\n=== HybridInputEmbedding Initialization ===")
            print(f"Model dimension (d_model): {d_model}")
            print(f"Continuous features: {d_continuous} -> {self.d_continuous_proj}")
            print(f"Categorical features: {n_categorical}")
            print(f"Categorical embedding dim per feature: {self.cat_embed_dim}")
            print(f"Total categorical dimensions: {self.d_categorical_proj}")
            print(f"Final dimensions: {self.d_continuous_proj + self.d_categorical_proj} (should equal {d_model})")
        
        # Verify dimensions add up correctly
        assert self.d_continuous_proj + self.d_categorical_proj == d_model, \
            f"Dimension mismatch: continuous ({self.d_continuous_proj}) + categorical ({self.d_categorical_proj}) != d_model ({d_model})"
        
        # Create layers
        # 1. Continuous projection: maps from d_continuous to d_continuous_proj
        self.continuous_projection = nn.Linear(d_continuous, self.d_continuous_proj)
        
        # 2. Categorical embeddings: one per categorical feature
        if n_categorical > 0:
            self.categorical_embeddings = nn.ModuleList([
                nn.Embedding(n_bins, self.cat_embed_dim)
                for _ in range(n_categorical)
            ])
        else:
            self.categorical_embeddings = None
        
        # 3. Final normalization (operating on d_model total dimensions)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, continuous_features: torch.Tensor, categorical_features: Optional[torch.Tensor] = None, debug: bool = False) -> torch.Tensor:
        """
        Process both continuous and categorical features through the hybrid embedding layer.
        
        Input Shapes:
            continuous_features: [batch_size, seq_len, d_continuous]
                - batch_size: Number of samples in the batch (e.g., 64)
                - seq_len: Length of the time series (e.g., 56)
                - d_continuous: Number of continuous features (e.g., 21)
            
            categorical_features: [batch_size, seq_len, n_categorical]
                - n_categorical: Number of categorical features (e.g., 10)
                - Values are indices into embedding tables
        
        Processing Steps:
        1. Continuous Features:
           - Project from d_continuous (21) to d_continuous_proj (~322)
           - This expands the feature space for better representation
        
        2. Categorical Features:
           - Each feature is embedded from a single index to cat_embed_dim (~10)
           - Total categorical dims = n_categorical (10) * cat_embed_dim (~10) ≈ 102
        
        3. Combined Output:
           - Concatenate continuous (322) and categorical (102) features
           - Final dimension = 424 (matches d_model)
           
        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = continuous_features.shape
        
        if debug:
            print(f"\n=== HybridInputEmbedding Forward Pass ===")
            print(f"Input continuous shape: {continuous_features.shape}")
            if categorical_features is not None:
                print(f"Input categorical shape: {categorical_features.shape}")
        
        # 1. Process continuous features
        # Reshape to [batch_size * seq_len, d_continuous] for efficient batch processing
        x_continuous = continuous_features.reshape(-1, continuous_features.size(-1))
        # Project to higher dimensional space [batch_size * seq_len, d_continuous_proj]
        x_continuous = self.continuous_projection(x_continuous)
        # Restore batch and sequence dimensions [batch_size, seq_len, d_continuous_proj]
        x_continuous = x_continuous.reshape(batch_size, seq_len, -1)
        
        if debug:
            print(f"Reshaped continuous after projection: {x_continuous.shape}")
        
        # 2. Process categorical features if present
        if categorical_features is not None:
            # Remove singleton dimension [batch_size, seq_len, 1] -> [batch_size, seq_len]
            if categorical_features.dim() == 3 and categorical_features.size(-1) == 1:
                categorical_features = categorical_features.squeeze(-1)
            if debug:
                print(f"Categorical indices shape: {categorical_features.shape}")
            
            # Process each categorical feature through its own embedding layer
            x_categoricals = []
            for i, embedding_layer in enumerate(self.categorical_embeddings):
                # Convert indices to embeddings [batch_size, seq_len] -> [batch_size, seq_len, cat_embed_dim]
                if categorical_features.dim() == 2:
                    x_cat = embedding_layer(categorical_features)
                else:
                    x_cat = embedding_layer(categorical_features[..., i])
                x_categoricals.append(x_cat)
            
            # Combine all categorical embeddings along feature dimension
            # Shape: [batch_size, seq_len, n_categorical * cat_embed_dim]
            x_categorical = torch.cat(x_categoricals, dim=-1)
            if debug:
                print(f"Categorical embedding shape: {x_categorical.shape}")
            
            # Combine continuous and categorical features
            # Shape: [batch_size, seq_len, d_continuous_proj + (n_categorical * cat_embed_dim)]
            x = torch.cat([x_continuous, x_categorical], dim=-1)
            if debug:
                print(f"After concatenation shape: {x.shape}")
                print(f"Expected final dimension: {self.d_continuous_proj + self.d_categorical_proj}")
        else:
            x = x_continuous
        
        # Verify dimensions match exactly
        expected_dim = self.d_continuous_proj + (self.d_categorical_proj if categorical_features is not None else 0)
        assert x.size(-1) == expected_dim, \
            f"Expected feature dim {expected_dim}, got {x.size(-1)}"
        
        # Apply normalization and dropout
        x = self.dropout(x)
        x = self.layer_norm(x)
        
        if debug:
            print(f"Final output shape: {x.shape}\n")
        
        return x

class TimeSeriesDecoder(nn.Module):
    def __init__(
        self,
        d_continuous: int = 2,
        n_categorical: int = 1,  # Number of categorical features
        n_bins: int = 10,  # Number of bins for categorical features
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_continuous_outputs: int = 2,  # Renamed from n_outputs
        n_categorical_outputs: Optional[int] = None,  # Number of categorical features to predict
        use_multi_scale: bool = False,
        use_relative_pos: bool = False,
        use_hope_pos: bool = False,
        temporal_scales: List[int] = [1, 2, 4],
        base: int = 10000
    ):
        super().__init__()
        
        # Store prediction dimensions
        self.n_continuous_outputs = n_continuous_outputs
        self.n_categorical_outputs = n_categorical_outputs
        self.n_bins = n_bins
        
        # Replace simple input projection with hybrid embedding
        self.input_embedding = HybridInputEmbedding(
            d_continuous=d_continuous,
            n_categorical=n_categorical,
            n_bins=n_bins,
            d_model=d_model,
            dropout=dropout
        )
        
        # Calculate how many unique blocks we need
        n_unique_blocks = (n_layers + 3) // 2
        
        # Create the unique decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model, 
                n_heads, 
                d_ff, 
                scales=temporal_scales if use_multi_scale else [1],
                dropout=dropout,
                use_relative_pos=use_relative_pos,
                use_hope_pos=use_hope_pos,
                base=base
            )
            for _ in range(n_unique_blocks)
        ])
        
        # Create a mapping of which block to use at each layer
        self.layer_mapping = self._create_layer_mapping(n_layers)
        
        self.ln_f = nn.LayerNorm(d_model)
        
        # Separate projection heads for continuous and categorical outputs
        self.continuous_projection = nn.Linear(d_model, n_continuous_outputs)
        
        # Create categorical projection heads if needed
        if n_categorical_outputs:
            self.categorical_projections = nn.ModuleList([
                nn.Linear(d_model, n_bins)  # One projection per categorical feature
                for _ in range(n_categorical_outputs)
            ])
        else:
            self.categorical_projections = None
        
        self.dropout = nn.Dropout(dropout)
    
    def _create_layer_mapping(self, n_layers: int) -> List[int]:
        """Creates a mapping of which block to use at each layer position."""
        # First and last blocks are always unique
        n_unique_blocks = (n_layers + 3) // 2
        if n_layers <= 2:
            return list(range(n_layers))
        
        # For layers in between, alternate between blocks
        mapping = [0]  # First block
        for i in range(1, n_layers-1):
            block_idx = 1 + (i-1) % (n_unique_blocks-2)  # Alternate between blocks 1 to n-2
            mapping.append(block_idx)
        mapping.append(n_unique_blocks-1)  # Last block
        
        return mapping
    
    def forward(self, continuous_features: torch.Tensor, categorical_features: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through the TimeSeriesDecoder.
        
        Input Shapes:
            continuous_features: [batch_size, seq_len, d_continuous]
            categorical_features: [batch_size, seq_len, n_categorical] (optional)
            
        Processing Steps:
        1. Embed input features to dimension d_model
        2. Apply transformer blocks for temporal processing
        3. Apply final layer normalization
        4. Project to output dimensions:
           - Continuous predictions
           - Categorical predictions (if enabled)
        5. Select only the final timestep predictions
        
        Returns:
            If n_categorical_outputs is None:
                Tensor of shape [batch_size, n_continuous_outputs] for continuous predictions
            Else:
                Tuple of:
                - Tensor of shape [batch_size, n_continuous_outputs] for continuous predictions
                - List of n_categorical_outputs tensors, each of shape [batch_size, n_bins] 
                  containing logits for categorical predictions
        """
        # Embed input features
        x = self.input_embedding(continuous_features, categorical_features)
        
        # Apply transformer blocks
        for block_idx in self.layer_mapping:
            x = self.blocks[block_idx](x)
        
        # Final processing
        x = self.ln_f(x)
        
        # Get final timestep features
        x = x[:, -1, :]  # Shape: [batch_size, d_model]
        
        # Generate continuous predictions
        continuous_out = self.continuous_projection(x)  # Shape: [batch_size, n_continuous_outputs]
        
        # Generate categorical predictions if enabled
        if self.categorical_projections is not None:
            categorical_out = [
                proj(x)  # Shape: [batch_size, n_bins]
                for proj in self.categorical_projections
            ]
            return continuous_out, categorical_out
        
        return continuous_out
