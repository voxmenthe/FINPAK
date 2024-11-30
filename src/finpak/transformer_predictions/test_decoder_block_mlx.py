import mlx.core as mx
import numpy as np
from timeseries_decoder_v4_mlx import DecoderBlockMLX

def test_decoder_block():
    """Test the MLX implementation of DecoderBlock."""
    
    # Initialize parameters
    batch_size = 32
    seq_len = 60
    d_model = 64
    n_heads = 4
    d_ff = 256
    scales = [1, 2, 4]
    
    print("\n=== Testing DecoderBlock MLX Implementation ===")
    
    # Create instance
    decoder = DecoderBlockMLX(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        scales=scales,
        dropout=0.1
    )
    
    # Test 1: Basic shape test
    print("\nTest 1: Input/Output shape verification")
    x = mx.random.normal((batch_size, seq_len, d_model))
    output = decoder(x)
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
        output = decoder(x)
        print(f"Sequence length {length} -> Output shape: {output.shape}")
        assert output.shape == (batch_size, length, d_model)
    
    # Test 4: Verify residual connections
    print("\nTest 4: Residual connection verification")
    x = mx.random.normal((batch_size, seq_len, d_model))
    output = decoder(x)
    residual_diff = mx.mean(mx.abs(output - x))
    print(f"Mean absolute difference from input: {residual_diff}")
    assert residual_diff > 1e-6, "Output is identical to input, residual connection might be broken"
    
    # Test 5: Layer normalization effect
    print("\nTest 5: Layer normalization verification")
    x = mx.random.normal((batch_size, seq_len, d_model))
    normalized = decoder.ln1(x)
    print(f"Layer norm output stats:")
    print(f"Mean: {mx.mean(normalized):.6f}")
    print(f"Std: {mx.std(normalized):.6f}")
    assert abs(mx.mean(normalized)) < 1e-5, "LayerNorm mean should be close to 0"
    assert abs(mx.std(normalized) - 1.0) < 1e-1, "LayerNorm std should be close to 1"
    
    print("\nAll tests passed successfully! ✅")

if __name__ == "__main__":
    try:
        test_decoder_block()
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise 