import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

from modern_timeseries_decoder import ModernDecoderBlock, TimeSeriesConfig
from fourier_head import Fourier_Head

# Assuming other necessary imports are already done

class MF_TimeseriesDecoder(nn.Module):
  """Modern Time Series Decoder using Fourier Head."""
  
  def __init__(
      self,
      d_input: int = 3,
      d_model: int = 64,
      n_heads: int = 4,
      n_layers: int = 3,
      num_frequencies: int = 10,
      dim_output: int = 100,  # Number of bins for discretization
      dropout: float = 0.1,
  ):
      super().__init__()
      
      # Input projection
      self.input_projection = nn.Linear(d_input, d_model, bias=False)
      
      # Stack of modern decoder blocks
      self.blocks = nn.ModuleList([
          ModernDecoderBlock(TimeSeriesConfig(
              d_input=d_input,
              d_model=d_model,
              n_heads=n_heads,
              n_layers=n_layers,
              dropout=dropout,
              n_outputs=dim_output  # Adjusted to match output dimension
          )) for _ in range(n_layers)
      ])
      
      # Output head
      self.ln_f = nn.LayerNorm(d_model)
      
      # Fourier Head
      self.fourier_head = Fourier_Head(
          dim_input=d_model,
          dim_output=dim_output,
          num_frequencies=num_frequencies
      )
      
      self.dropout = nn.Dropout(dropout)
      
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      B, T, C = x.shape
      
      # Project input
      x = self.input_projection(x)
      x = F.layer_norm(x, (x.size(-1),))
      
      x0 = x
      v1 = None
      
      # Pass through decoder blocks
      for block in self.blocks:
          x, v1 = block(x, v1, x0)
      
      # Final layernorm
      x = F.layer_norm(x, (x.size(-1),))
      
      # Use only the last position for prediction
      x = x[:, -1, :]  # Shape: (B, d_model)
      
      # Pass through Fourier Head
      x = self.fourier_head(x)
      
      return x  # Logits over bins