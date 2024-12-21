import torch
import numpy as np
from finpak.transformer_predictions.timeseries_decoder_v4 import HoPE, DecoderBlock, TimeSeriesDecoder

# GLOBAL PARAMETERS
MODEL_CONFIG = {
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 3,
    'd_ff': 256,
    'dropout': 0.1,
    'batch_size': 32,
    'seq_len': 16,
    'd_input': 3,
    'n_outputs': 2,
    'use_multi_scale': False,
    'scales': [1, 2, 4]
}

def test_hope():
    """Test the Hybrid Positional Encoding (HoPE) module."""
    print("Testing HoPE...")
    hope = HoPE(d_model=MODEL_CONFIG['d_model'])
    # Forward pass
    pos_enc = hope(MODEL_CONFIG['seq_len'])
    print(f"  HoPE output shape: {pos_enc.shape}")
    # Some diagnostics
    print(f"  Number of active components (from internal calc): {hope._calculate_active_components(MODEL_CONFIG['seq_len'])}")
    print(f"  First element mean: {pos_enc[0,0].mean().item():.4f}, std: {pos_enc[0,0].std().item():.4f}")

def test_decoder_block():
    """Test a single decoder block with attention and feed-forward layers."""
    print("Testing DecoderBlock...")
    block = DecoderBlock(
        d_model=MODEL_CONFIG['d_model'], 
        n_heads=MODEL_CONFIG['n_heads'], 
        d_ff=MODEL_CONFIG['d_ff'],
        scales=[1]  # Use single scale to avoid dimension changes
    )
    x = torch.randn(MODEL_CONFIG['batch_size'], MODEL_CONFIG['seq_len'], MODEL_CONFIG['d_model'])
    out = block(x)
    print(f"  DecoderBlock output shape: {out.shape}")
    # Some diagnostics
    diff = (out - x).abs().mean().item()
    print(f"  Mean absolute difference between input and output: {diff:.4f}")

def test_time_series_decoder():
    """Test the complete TimeSeriesDecoder model."""
    print("Testing TimeSeriesDecoder...")
    model = TimeSeriesDecoder(
        d_input=MODEL_CONFIG['d_input'],
        d_model=MODEL_CONFIG['d_model'],
        n_heads=MODEL_CONFIG['n_heads'],
        n_layers=MODEL_CONFIG['n_layers'],
        d_ff=MODEL_CONFIG['d_ff'],
        dropout=MODEL_CONFIG['dropout'],
        n_outputs=MODEL_CONFIG['n_outputs'],
        use_multi_scale=MODEL_CONFIG['use_multi_scale'],
        temporal_scales=MODEL_CONFIG['scales']
    )
    
    x = torch.randn(MODEL_CONFIG['batch_size'], MODEL_CONFIG['seq_len'], MODEL_CONFIG['d_input'])
    out = model(x)
    print(f"  TimeSeriesDecoder output shape: {out.shape}")
    
    # Diagnostics
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {param_count}")
    print(f"  Output example (first item): {out[0].detach().numpy()}")

def main():
    """Run all tests sequentially."""
    test_hope()
    print()
    test_decoder_block()
    print()
    test_time_series_decoder()

if __name__ == "__main__":
    main()
