import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from typing import List

#########################################################
"""
Below is a single, self-contained “TimeSeriesDecoder” implementation that includes:
• The HoPE rotary-like positional encoding.
• Multi-scale attention if desired.
• Optional “learned residual connection” on the Value vector (v) by blending the newly projected v with an external/previous v1, controlled by lamb1/lamb2 (like in the “speedrun” code).
• MixLayerNorm logic for pre-LN vs. post-LN.
• A demonstration of how to pass v1 between layers.
This version borrows from all the improvements we’ve previously discussed, plus the v1 gating from the “speedrun” code. Adapt it as needed (e.g., if you don’t need multi-scale or the “v1” gating at every block).
────────────────────────────────────────────────────────────────────────

HoPE: High-frequency Rotary Position Encoding
────────────────────────────────────────────────────────────────────────
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


class HoPE(nn.Module):
    """
    High-frequency rotary Position Encoding (HoPE).  
    This version dynamically determines how many high-frequency components
    to keep based on the sequence length, then uses random param 'position_independent'
    for the remainder (low-frequency) dimensions.
    """
    def __init__(self, d_model: int, base: int = 10000):
        super().__init__()
        assert d_model % 2 == 0, "Model dimension must be divisible by 2"
        self.d_model = d_model
        self.base = base

        # Up to 1/4 of dims can be purely position-independent
        self.max_position_independent = d_model // 4
        self.position_independent = nn.Parameter(torch.randn(self.max_position_independent, 2))

    def _calculate_active_components(self, seq_len: int) -> int:
        dim = self.d_model // 2
        positions = torch.arange(dim, dtype=torch.float32, device=self.position_independent.device)
        freqs = 1.0 / (self.base ** (positions / dim))
        # Keep only those freq_i where freq_i * 2π * seq_len >= 1
        condition = (freqs * 2 * math.pi * seq_len) >= 1
        active_comp = int(condition.sum().item())
        return min(active_comp, dim - self.max_position_independent)

    def _get_freq_bands(self, active_comp_id: int) -> torch.Tensor:
        dim = self.d_model // 2
        positions = torch.arange(dim, dtype=torch.float32, device=self.position_independent.device)
        freqs = 1.0 / (self.base ** (positions / dim))
        return freqs[:active_comp_id].unsqueeze(0)  # [1, active_comp_id]

    def forward(self, seq_len: int) -> torch.Tensor:
        # Decide how many “high-frequency” components to keep
        active_comp_id = self._calculate_active_components(seq_len)
        freq_bands = self._get_freq_bands(active_comp_id)
        device = self.position_independent.device

        # Create a matrix of relative positions [seq_len, seq_len]
        positions = torch.arange(seq_len, device=device)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # [seq_len, seq_len]

        # Multiply by freq bands => (seq_len, seq_len, active_comp_id)
        rel_pos_expanded = rel_pos.unsqueeze(-1)
        theta = rel_pos_expanded * freq_bands

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        # Stack them => shape [seq_len, seq_len, active_comp_id, 2]
        high_freq = torch.stack([cos_theta, sin_theta], dim=-1)

        # Position-independent remainder
        latter_dim = (self.d_model // 2) - active_comp_id
        pos_independent = self.position_independent[:latter_dim]
        pos_independent = pos_independent.unsqueeze(0).unsqueeze(0)  # [1,1,latter_dim,2]
        pos_independent = pos_independent.expand(seq_len, seq_len, -1, -1)

        combined = torch.cat([high_freq, pos_independent], dim=2)  # shape [seq_len, seq_len, d_model//2, 2]
        return combined.reshape(seq_len, seq_len, self.d_model)

# ────────────────────────────────────────────────────────────────────────
# 2) MixLayerNorm: “Pre-LN” vs. “Post-LN” + optional toggles
# ────────────────────────────────────────────────────────────────────────
class MixLayerNorm(nn.Module):
    """
    Mix of pre-LayerNorm and post-LayerNorm, based on a flag.
    You can expand this to add RMS Norm if desired.
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

# ────────────────────────────────────────────────────────────────────────
# 3) CausalSelfAttention with Optional “v1” Gating
# ────────────────────────────────────────────────────────────────────────
# Below is the heart of the GPT-inspired “learned residual connection” on the Value vector. We add lamb1 and lamb2 parameters to blend the newly projected v with an external v1.
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, base: int = 10000):
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

        # For causal masking (1024 is arbitrary max seq length you might handle)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(1, 1, 1024, 1024), diagonal=1).bool(),
            persistent=False
        )

        self.pos_enc = HoPE(d_model, base=base)

    def forward(self, x: torch.Tensor, v1: torch.Tensor=None):
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

# ────────────────────────────────────────────────────────────────────────
# 4) MultiScaleAttention (Optional)
# ────────────────────────────────────────────────────────────────────────
# If you want multi-scale attention, we must decide how “v1” gating applies across multiple scales. Below, we show a simplistic approach where we reuse the same v1 for each scale and return only the last scale’s v as new_v. You can refine if you want to track a separate “v1” per scale.
class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        scales: List[int] = [1, 2, 4],
        dropout: float = 0.1,
        base: int = 10000
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scales = scales

        self.attention_modules = nn.ModuleList([
            CausalSelfAttention(d_model, n_heads, dropout=dropout, base=base)
            for _ in scales
        ])
        self.scale_combine = nn.Linear(len(scales) * d_model, d_model)

    def forward(self, x: torch.Tensor, v1: torch.Tensor=None):
        """
        x: Shape [B, T, d_model]
        v1: shape [B, T, n_heads, head_dim], if you're carrying it across blocks

        Returns:
          combined_out: [B, T, d_model]
          new_v:        [B, T, n_heads, head_dim]
        """
        B, T, C = x.shape
        scale_outputs = []
        new_v_global = None

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

# ────────────────────────────────────────────────────────────────────────
# 5) DecoderBlock: MixLayerNorm + ReLU² MLP + v1 Pipeline
# ────────────────────────────────────────────────────────────────────────
# We integrate the gating approach by receiving and returning “v1.” The “v1” can be threaded from layer to layer at your discretion.
class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        scales: List[int] = [1, 2, 4],
        dropout: float = 0.1,
        mixln_alpha: float = 0.25,  # fraction of layers that do post-LN vs. pre-LN
        base: int = 10000
    ):
        super().__init__()
        self.d_model = d_model
        self.mixln_alpha = mixln_alpha

        self.ln1 = MixLayerNorm(d_model)
        # Multi-scale or single-scale attention
        self.attention = MultiScaleAttention(d_model, n_heads, scales, dropout, base)
        self.ln2 = MixLayerNorm(d_model)

        # GPT-like MLP with ReLU²
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout),
            nn.Lambda(lambda t: F.relu(t).square()),  # ReLU²
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

# ────────────────────────────────────────────────────────────────────────
# 6) TimeSeriesDecoder: Stacking Blocks & Threading “v1”
# ────────────────────────────────────────────────────────────────────────
# Finally, the main decoder that arranges blocks in the order you specified, plus your layer-mapping approach (or a simpler “one block per layer” approach). Each call to a block returns (x, new_v), which we feed forward to the next layer.
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
        mixln_alpha: float = 0.25,  # fraction of layers that do post-LN
        base: int = 10000
    ):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(d_input, d_model)
        self.n_layers = n_layers

        # Decide how many unique blocks, or just let it match n_layers.
        # If you want your special layer mapping, do it here:
        n_unique_blocks = (n_layers + 3) // 2

        # Build unique blocks
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

        # Map each layer index to one of the unique blocks
        self.layer_mapping = self._create_layer_mapping(n_layers)

        # Final layer norm + output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, n_outputs)
        self.dropout = nn.Dropout(dropout)

    def _create_layer_mapping(self, n_layers: int) -> list:
        """
        Example logic from previous code:
          - The first and last blocks are unique.
          - Middle layers may be repeated in pairs or triples.
        """
        mapping = [0]
        next_unique_idx = 1
        middle_layers = n_layers - 2

        if middle_layers <= 0:
            mapping.append(next_unique_idx)
            return mapping

        if middle_layers % 3 == 0:
            triple_sets = middle_layers // 3
            triple_blocks = [next_unique_idx, next_unique_idx + 1, next_unique_idx + 2]
            for _ in range(triple_sets):
                mapping.extend(triple_blocks)
            next_unique_idx += 3
        elif middle_layers % 2 == 0:
            pair_sets = middle_layers // 2
            pair_blocks = [next_unique_idx, next_unique_idx + 1]
            for _ in range(pair_sets):
                mapping.extend(pair_blocks)
            next_unique_idx += 2
        else:
            # Mixed approach
            triples_to_use = middle_layers // 3
            remaining = middle_layers % 3
            if triples_to_use > 0:
                triple_blocks = [next_unique_idx, next_unique_idx + 1, next_unique_idx + 2]
                for _ in range(triples_to_use):
                    mapping.extend(triple_blocks)
                next_unique_idx += 3
            if remaining >= 2:
                pair_blocks = [next_unique_idx, next_unique_idx + 1]
                mapping.extend(pair_blocks)
                next_unique_idx += 2
                remaining -= 2
            # leftover single layers
            while remaining > 0:
                mapping.append(next_unique_idx)
                next_unique_idx += 1
                remaining -= 1

        # final unique block
        mapping.append(next_unique_idx)
        return mapping

    def forward(self, x: torch.Tensor):
        """
        x: shape [B, T, d_input]
        Returns [B, n_outputs]
        """
        B, T, _ = x.shape

        # Project input
        x = self.input_projection(x)
        x = self.dropout(x)

        # Go through each layer in sequence, passing v1 forward
        v1 = None
        for i, block_idx in enumerate(self.layer_mapping):
            x, new_v = self.blocks[block_idx](x, i, self.n_layers, v1)
            v1 = new_v  # carry v forward for gating in the next layer

        # Use the last token for final prediction
        x = self.ln_f(x[:, -1])  # [B, d_model]
        return self.output_projection(x)
"""
────────────────────────────────────────────────────────────────────────
7) How to Use This “v1” Mechanism
────────────────────────────────────────────────────────────────────────
• By default, if you never pass an initial v1, each layer will just treat v1=None, so “v” gating effectively becomes a no-op in the first layer (since v1 defaults to v).
• However, after the first layer, each subsequent layer receives the “new_v” from the previous block. This allows the model to “blend” the newly computed value with the old one.
• If you prefer not to chain “v1” across every layer, you could simply omit it in MultiScaleAttention or store multiple v1’s. The above code is just a straightforward example.
────────────────────────────────────────────────────────────────────────
8) Summary
────────────────────────────────────────────────────────────────────────
This final code merges:
• HoPE positional encoding.
• MixLayerNorm logic with partial pre/post LN.
• Multi-scale attention (optional) that calls a “CausalSelfAttention.”
• A learned residual/gating on the Value vector (v) via lamb1 and lamb2 (like the “speedrun” code’s v1 approach).
• GPT-like MLP with ReLU².
You can now train/test this updated TimeSeriesDecoder architecture. If your use-case doesn’t need multi-pass gating, you could remove “v1” altogether or keep it as an experimental feature.
"""
#########################################################


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

        self.lamb1 = nn.Parameter(torch.tensor(0.5))
        self.lamb2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, kv_cache=None, freq=None, v1=None):
        B, T, C = x.size()  # if this is sampling, T would be 1.

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)  # B, T, n_head, D
        cos, sin = freq

        if v1 is None:
            v1 = v

        v = self.lamb1 * v + self.lamb2 * v1.view_as(v)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache

            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))

            if k_cache is not None:
                if isinstance(k_cache, int):
                    k_cache = k
                    v_cache = v
                else:
                    k = torch.cat([k_cache, k], dim=1)
                    v = torch.cat([v_cache, v], dim=1)  # it cats in T dim.

                new_kv_cache = (k, v)

            # do classic attention.
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
            )

        else:

            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
            new_kv_cache = None
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
            )

        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return (y, v1), new_kv_cache





















#########################################################
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

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(1, 1, 1024, 1024), diagonal=1).bool()
        )
        self.pos_enc = HoPE(d_model, base=base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        pos_bias = self.pos_enc(T)
        pos_bias = pos_bias.view(T, T, self.n_heads, self.head_dim)
        pos_bias = pos_bias.sum(dim=-1)
        pos_bias = pos_bias.permute(2, 0, 1).unsqueeze(0)
        att = att + pos_bias
        att = att.masked_fill(self.mask[:, :, :T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        return out

class MultiScaleAttention(nn.Module):
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
        
        self.attention_modules = nn.ModuleList([
            CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, bias=bias, base=base)
            for _ in scales
        ])
        
        self.scale_combine = nn.Linear(len(scales) * d_model, d_model, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        scale_outputs = []
        for scale, attention in zip(self.scales, self.attention_modules):
            if scale > 1:
                x_scaled = F.avg_pool1d(
                    x.transpose(1, 2), 
                    kernel_size=scale, 
                    stride=scale
                ).transpose(1, 2)
                attended = attention(x_scaled)
                attended = F.interpolate(
                    attended.transpose(1, 2),
                    size=T,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            else:
                attended = attention(x)
            scale_outputs.append(attended)
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
        alpha: float = 0.25,  # Mix-LN hyperparameter
        base: int = 10000,
    ):
        super().__init__()
        self.d_model = d_model
        self.alpha = alpha
        
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
        if layer_idx < int(self.alpha * total_layers):
            # Post-LN
            x_norm = self.ln1.forward(x + self.dropout(self.attention(x)), is_pre_ln=False)
            x = x_norm + self.dropout(self.mlp(x_norm))
            return x
        else:
            # Pre-LN
            x_norm = self.ln1.forward(x, is_pre_ln=True)
            x = x + self.dropout(self.attention(x_norm))
            x_norm = self.ln2.forward(x, is_pre_ln=True)
            x = x + self.dropout(self.mlp(x_norm))
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
        alpha: float = 0.25, # Mix-LN hyperparameter
        base: int = 10000
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(d_input, d_model)
        self.n_layers = n_layers # Store the number of layers
        
        n_unique_blocks = (n_layers + 3) // 2
        
        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model, 
                n_heads, 
                d_ff, 
                scales=temporal_scales if use_multi_scale else [1],
                dropout=dropout,
                alpha=alpha,
                base=base
            )
            for _ in range(n_unique_blocks)
        ])
        
        self.layer_mapping = self._create_layer_mapping(n_layers)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, n_outputs)
        self.dropout = nn.Dropout(dropout)
        
    def _create_layer_mapping(self, n_layers: int) -> list:
        mapping = [0]
        next_unique_idx = 1
        middle_layers = n_layers - 2
        
        if middle_layers <= 0:
            mapping.append(next_unique_idx)
            return mapping
        
        if middle_layers % 3 == 0:
            triple_sets = middle_layers // 3
            triple_blocks = [next_unique_idx, next_unique_idx + 1, next_unique_idx + 2]
            for _ in range(triple_sets):
                mapping.extend(triple_blocks)
            next_unique_idx += 3
            
        elif middle_layers % 2 == 0:
            pair_sets = middle_layers // 2
            pair_blocks = [next_unique_idx, next_unique_idx + 1]
            for _ in range(pair_sets):
                mapping.extend(pair_blocks)
            next_unique_idx += 2
            
        else:
            triples_to_use = middle_layers // 3
            remaining = middle_layers % 3
            
            if triples_to_use > 0:
                triple_blocks = [next_unique_idx, next_unique_idx + 1, next_unique_idx + 2]
                for _ in range(triples_to_use):
                    mapping.extend(triple_blocks)
                next_unique_idx += 3
            
            if remaining >= 2:
                pair_blocks = [next_unique_idx, next_unique_idx + 1]
                mapping.extend(pair_blocks)
                next_unique_idx += 2
                remaining -= 2
            
            while remaining > 0:
                mapping.append(next_unique_idx)
                next_unique_idx += 1
                remaining -= 1
        
        mapping.append(next_unique_idx)
        
        return mapping
        
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        x = self.input_projection(x)
        x = self.dropout(x)
        
        original_x = x
        for i, block_idx in enumerate(self.layer_mapping):
            x = self.blocks[block_idx].forward(x, i, self.n_layers) + original_x
            original_x = x
        
        x = self.ln_f(x[:, -1])
        return self.output_projection(x)

if __name__ == '__main__':
    # Example usage
    input_tensor = torch.randn(16, 20, 3)  # Batch size 16, sequence length 20, 3 input features
    model = TimeSeriesDecoder(n_layers=4, use_multi_scale=True)
    output = model(input_tensor)
    print(output.shape)