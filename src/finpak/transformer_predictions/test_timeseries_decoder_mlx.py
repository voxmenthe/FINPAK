import mlx.core as mx
import numpy as np
from timeseries_decoder_v4_mlx import TimeseriesDecoderMLX

def test_timeseries_decoder():
    """Test the MLX implementation of TimeseriesDecoder."""
    
    # Initialize parameters
    batch_size = 32
    seq_len = 60
    d_input = 3
    d_model = 64
    n_heads = 4
    n_outputs = 2
    
    print("\n=== Testing TimeseriesDecoder MLX Implementation ===")
    
    # Create instance
    decoder = TimeseriesDecoderMLX(
        d_input=d_input,
        d_model=d_model,
        n_heads=n_heads,
        n_outputs=n_outputs,
        use_multi_scale=True
    )
    
    # Test 1: Basic shape test
    print("\nTest 1: Input/Output shape verification")
    x = mx.random.normal((batch_size, seq_len, d_input))
    output = decoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, n_outputs), \
        f"Expected shape {(batch_size, n_outputs)}, got {output.shape}"
    
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
        x = mx.random.normal((batch_size, length, d_input))
        output = decoder(x)
        print(f"Sequence length {length} -> Output shape: {output.shape}")
        assert output.shape == (batch_size, n_outputs)
    
    print("\nAll tests passed successfully! ✅")

if __name__ == "__main__":
    try:
        test_timeseries_decoder()
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise 