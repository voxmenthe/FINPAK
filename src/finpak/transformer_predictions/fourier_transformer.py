import torch
import torch.nn as nn
from fourier_head import Fourier_Head  # Note the underscore in the original implementation
from timeseries_decoder import DecoderBlock


class TimeSeriesDecoderWithFourierHead(nn.Module):
    def __init__(
        self,
        d_input: int = 3,          # Number of input features (e.g., open, high, low, close)
        d_model: int = 64,         # Hidden dimension
        n_heads: int = 4,          # Number of attention heads
        n_layers: int = 3,         # Number of decoder layers
        d_ff: int = 256,          # Feed-forward dimension
        dropout: float = 0.1,
        n_bins: int = 100,        # Number of price bins for discretization
        n_frequencies: int = 32,   # Number of Fourier frequencies
        device: str = "cuda"
    ):
        super().__init__()
        
        # Input projection and positional embedding
        self.input_projection = nn.Linear(d_input, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1024, d_model))
        
        # Stack of decoder blocks (unchanged)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        
        # Replace linear output projection with Fourier head
        self.fourier_head = Fourier_Head(
            dim_input=d_model,
            dim_output=n_bins,  # Number of price bins
            num_frequencies=n_frequencies,
            regularizion_gamma=1e-6,  # Small regularization to prevent overfitting
            device=device
        )
        
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
            
        # Get final hidden state
        x = self.ln_f(x[:, -1])  # Shape: [batch_size, d_model]
        
        # Pass through Fourier head to get price distribution
        # Returns log probabilities over price bins
        return self.fourier_head(x)

class StockPricePredictor:
    def __init__(self, model, price_range, n_bins):
        self.model = model
        self.min_price, self.max_price = price_range
        self.n_bins = n_bins
        self.bin_edges = torch.linspace(self.min_price, self.max_price, n_bins + 1)
        
    def discretize_price(self, price):
        """Convert continuous price to bin index"""
        return torch.bucketize(price, self.bin_edges) - 1
        
    def get_continuous_prediction(self, logits):
        """Convert predicted distribution to expected price"""
        probs = torch.softmax(logits, dim=-1)
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        expected_price = (probs * bin_centers).sum(dim=-1)
        return expected_price