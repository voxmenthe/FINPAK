import mlx.core as mx
import numpy as np
from timeseries_decoder_v4_mlx import HoPEMLX

def test_hope_mlx():
    """Test the MLX implementation of HoPE with dummy data."""
    
    # Initialize parameters
    d_model = 64
    seq_len = 32
    
    # Create HoPE instance
    hope_mlx = HoPEMLX(
        d_model=d_model,
        max_position_independent=0,
        base=10000.0,
        theta_type="log",
        cumsum_pow=True
    )
    
    print("Test 1: Output shape verification")
    output = hope_mlx(seq_len)
    
    # Rest of tests...

def test_hope_dimensions():
    """Test HoPE encoding with various sequence lengths and model dimensions."""
    print("\nTesting HoPE dimensions:")
    
    test_configs = [
        (64, 32),   # (d_model, seq_len)
        (128, 64),
        (256, 128)
    ]
    
    for d_model, seq_len in test_configs:
        print(f"\nTesting d_model={d_model}, seq_len={seq_len}")
        hope = HoPEMLX(d_model=d_model)
        
        # Get active components
        active_comp_id = hope._calculate_active_components(seq_len)
        latter_dim = max(0, (d_model // 2) - active_comp_id)
        
        print(f"Active components: {active_comp_id}")
        print(f"Latter dim: {latter_dim}")
        print(f"Position independent shape: {hope.position_independent.shape}")
        
        # Generate encodings
        encodings = hope(seq_len)
        print(f"Output shape: {encodings.shape}")
        assert encodings.shape == (seq_len, seq_len, d_model)

if __name__ == "__main__":
    print("Testing HoPE MLX:")
    try:
        test_hope_mlx()
        test_hope_dimensions()
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise 