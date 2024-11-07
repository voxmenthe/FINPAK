import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import warnings


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        max_sequence_length: int = 1024,
        max_cache_size: int = 10,
        warn_on_cache_full: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_sequence_length = max_sequence_length
        self.max_cache_size = max_cache_size
        self.warn_on_cache_full = warn_on_cache_full
        self._cache_full_warned = False
        
        # Single projection matrix for Q, K, V
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize mask cache
        self._mask_cache = {}
        
    def clear_mask_cache(self):
        """Clear the mask cache to free memory."""
        self._mask_cache.clear()

    def limit_mask_cache(self):
        """Limit the mask cache to the most recent sizes."""
        if len(self._mask_cache) > self.max_cache_size:
            if self.warn_on_cache_full and not self._cache_full_warned:
                warnings.warn(
                    f"Mask cache exceeded size {self.max_cache_size}. "
                    "Consider increasing max_cache_size if this happens frequently."
                )
                self._cache_full_warned = True
                
            # Keep only the most recently used sizes
            cache_items = sorted(
                self._mask_cache.items(),
                key=lambda x: x[1].last_used,
                reverse=True
            )[:self.max_cache_size]
            self._mask_cache = dict(cache_items)
    
    def _get_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Get cached causal mask or create a new one."""
        if seq_length not in self._mask_cache:
            # Create new mask if not in cache
            mask = torch.triu(
                torch.ones(seq_length, seq_length, device=device),
                diagonal=1
            ).bool()
            # Store mask with timestamp
            self._mask_cache[seq_length] = CacheMask(mask)
            # Limit cache size
            self.limit_mask_cache()
        else:
            # Ensure mask is on the correct device
            cached_mask = self._mask_cache[seq_length]
            if cached_mask.mask.device != device:
                cached_mask.mask = cached_mask.mask.to(device)
            cached_mask.update_timestamp()
            
        return self._mask_cache[seq_length].mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        if T > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {T} exceeds maximum length {self.max_sequence_length}"
            )
        
        # Calculate Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Get cached causal mask
        causal_mask = self._get_causal_mask(T, x.device)
        
        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        att = att.masked_fill(causal_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        # Combine heads
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        
        return out

class CacheMask:
    """Helper class to track mask usage."""
    def __init__(self, mask: torch.Tensor):
        self.mask = mask
        self.last_used = time.time()
    
    def update_timestamp(self):
        """Update the last used timestamp."""
        self.last_used = time.time()

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
        n_outputs: int = 2,
        max_sequence_length: int = 1024
    ):
        super().__init__()
        
        self.max_sequence_length = max_sequence_length
        self.input_projection = nn.Linear(d_input, d_model)
        
        # Update positional embedding to use max_sequence_length
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_sequence_length, d_model)
        )
        
        # Stack of decoder blocks with max_sequence_length
        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model,
                n_heads,
                d_ff,
                dropout
            ) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, n_outputs)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        if T > self.max_sequence_length:
            raise ValueError(
                f"Input sequence length {T} exceeds maximum length {self.max_sequence_length}"
            )
        
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
