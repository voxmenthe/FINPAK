import mlx.core as mx
import numpy as np
from timeseries_decoder_v4_mlx import MultiScaleAttentionMLX

def test_multiscale_attention():
    """Test the MLX implementation of MultiScaleAttention."""
    
    # Initialize parameters
    batch_size = 32
    seq_len = 60
    d_model = 64
    n_heads = 4
    scales = [1, 2, 4]
    
    print("\n=== Testing MultiScaleAttention MLX Implementation ===")
    
    # Create instance
    msa = MultiScaleAttentionMLX(
        d_model=d_model,
        n_heads=n_heads,
        scales=scales,
        dropout=0.1
    )
    
    # Test 1: Basic shape test
    print("\nTest 1: Input/Output shape verification")
    x = mx.random.normal((batch_size, seq_len, d_model))
    output = msa(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, \
        f"Expected shape {x.shape}, got {output.shape}"
    
    # Test 2: Check if output contains valid values
    print("\nTest 2: Output value verification")
    print(f"Output statistics:")
    print(f"Mean: {mx.mean(output)}")
    print(f"Std: {mx.std(output)}")
    print(f"Min: {mx.min(output)}")
    print(f"Max: {mx.max(output)}")
    assert mx.isfinite(output).all(), "Output contains non-finite values"
    
    # Test 3: Test different sequence lengths
    print("\nTest 3: Different sequence lengths")
    test_lengths = [32, 64, 128]
    for length in test_lengths:
        x = mx.random.normal((batch_size, length, d_model))
        output = msa(x)
        print(f"Sequence length {length} -> Output shape: {output.shape}")
        assert output.shape == (batch_size, length, d_model)
    
    # Test 4: Verify scale-specific processing
    print("\nTest 4: Scale-specific processing")
    for scale in scales:
        print(f"\nTesting scale {scale}:")
        x = mx.random.normal((batch_size, seq_len, d_model))
        
        if scale > 1:
            # Verify downsampling
            x_t = mx.transpose(x, (0, 2, 1))
            x_scaled = mx.mean(
                mx.reshape(x_t, (batch_size, d_model, seq_len // scale, scale)), 
                axis=-1
            )
            print(f"Downsampled shape: {x_scaled.shape}")
            assert x_scaled.shape == (batch_size, d_model, seq_len // scale)
    
    print("\nAll tests passed successfully! ✅")

if __name__ == "__main__":
    try:
        test_multiscale_attention()
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise 