import torch
from typing import List, Optional, Tuple, Union
from stock_dataset import StockFeatures
import numpy as np
from scipy import stats  # For Student-t distribution


def calculate_returns(prices: torch.Tensor, period: int, debug: bool = False) -> torch.Tensor:
    """Calculate percentage returns over specified periods"""
    if not isinstance(period, int):
        print(f"Debug: calculate_returns called with period type {type(period)}")
        print(f"Debug: period value: {period}")
        print(f"Debug: call stack:")
        import traceback
        traceback.print_stack()
        raise TypeError(f"period must be an integer, got {type(period)} instead")

    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    if debug:
        print(f"\nCalculating {period}-day returns:")
        print(f"First few prices: {prices[:5].tolist()}")
        print(f"Last few prices: {prices[-5:].tolist()}")
        print(f"Prices shape: {prices.shape}")

    # Ensure prices is 2D if it's not already
    if len(prices.shape) == 1:
        prices = prices.unsqueeze(-1)
    
    # Add small epsilon to denominator to prevent division by zero
    eps = 1e-8
    denominator = prices[:-period]
    # Handle negative prices by using absolute value for denominator
    # but preserving the sign in the final calculation
    denominator_sign = torch.sgn(denominator)
    denominator_abs = torch.abs(denominator) + eps
    numerator = prices[period:] - prices[:-period]
    returns = numerator / denominator_abs
    # Preserve the original sign
    returns = returns * denominator_sign

    # Debug: Print intermediate values for first few calculations
    if debug and len(returns) > 0:
        print(f"\nDetailed return calculations for first few points:")
        for i in range(min(3, len(returns))):
            end_price = prices[period + i].item()
            start_price = prices[i].item()
            calc_return = returns[i].item()
            print(f"Index {i}:")
            print(f"  Start price (t): {start_price:.4f}")
            print(f"  End price (t+{period}): {end_price:.4f}")
            print(f"  Return: {calc_return:.4f} ({calc_return*100:.2f}%)")

    # Replace infinite values with the maximum float value
    returns = torch.nan_to_num(returns, nan=0.0, posinf=3.4e38, neginf=-3.4e38)

    # Debug: Print some sample returns
    if debug:
        print(f"\nReturn statistics:")
        finite_returns = returns[torch.isfinite(returns)]
        if len(finite_returns) > 0:
            print(f"Mean: {finite_returns.mean().item():.4f}")
            print(f"Std: {finite_returns.std().item():.4f}")
            print(f"Min: {finite_returns.min().item():.4f}")
            print(f"Max: {finite_returns.max().item():.4f}")
            print(f"First few returns: {returns[:5].tolist()}")
            print(f"Last few returns: {returns[-5:].tolist()}")
        else:
            print("Warning: No finite returns found!")

    # Create padding with correct shape
    padding = torch.zeros((period,) + returns.shape[1:], dtype=prices.dtype, device=prices.device)
    padded_returns = torch.cat([padding, returns], dim=0)
    
    # If input was 1D, make output 1D
    if len(prices.shape) == 2 and prices.shape[1] == 1:
        padded_returns = padded_returns.squeeze(-1)
        
    if debug:
        print(f"\nFinal padded returns shape: {padded_returns.shape}")
        print(f"First few padded returns: {padded_returns[:period+5].tolist()}")
        
    return padded_returns

def calculate_sma(prices: torch.Tensor, window: int) -> torch.Tensor:
    """Calculate Simple Moving Average"""
    sma = torch.zeros_like(prices)
    for i in range(window, len(prices) + 1):
        sma[i-1] = prices[i-window:i].mean()
    return sma

def create_stock_features(
    prices: torch.Tensor,
    config: dict,
    debug: bool = False
) -> StockFeatures:
    """
    Create feature matrix and target variables from price series

    Returns:
        StockFeatures object containing:
        - features: Tensor of shape (n_samples, n_features)
        - targets: Tensor of shape (n_samples, n_targets)
        - feature_names: List of feature names
        - target_names: List of target names
        - valid_start_idx: Index where features become valid after initialization period
    """

    if debug:
        print("\n=== Feature Creation Debug ===")
        print(f"Input price tensor type: {prices.dtype}")
        print(f"Contains complex numbers: {torch.is_complex(prices)}")
        if torch.is_complex(prices):
            print("WARNING: Input prices contain complex numbers!")
            print(f"First complex number found at index: {torch.where(torch.is_complex(prices))[0][0].item()}")
            complex_prices = prices[torch.where(torch.is_complex(prices))[0][:5]]
            print(f"Sample complex values: {complex_prices.tolist()}")
        print(f"Input price stats: min={prices.real.min().item():.4f}, max={prices.real.max().item():.4f}")
        print(f"NaN in prices: {torch.isnan(prices).sum().item()}")

    # Extract parameters from config
    use_volatility = config['data_params']['use_volatility']
    use_momentum = config['data_params']['use_momentum']
    momentum_periods = config['data_params']['momentum_periods']
    return_periods = config['data_params']['return_periods']
    sma_periods = config['data_params']['sma_periods']
    target_periods = config['data_params']['target_periods']

    # Calculate the initialization period needed
    max_lookback = max([
        max(sma_periods) if sma_periods else 0,
        max(momentum_periods) if use_momentum else 0,
        max(return_periods) if use_volatility else 0,  # For volatility calculation
        max(return_periods),  # For return features
        max(target_periods)   # For target calculation
    ])

    if debug:
        print("\nCreating stock features:")
        print(f"Price series length: {len(prices)}")
        print(f"Required initialization period: {max_lookback}")
        print(f"First few prices: {prices[:5].tolist()}")

    # Ensure we have enough data points
    if len(prices) <= max_lookback:
        raise ValueError(f"Price series length ({len(prices)}) must be greater than maximum lookback period ({max_lookback})")

    feature_list = []
    feature_names = []

    # Calculate return features
    for period in return_periods:
        returns = calculate_returns(prices, period)
        if debug:
            print(f"\n{period}-day returns:")
            print(f"Shape: {returns.shape}")
            print(f"Stats: min={returns.min().item():.4f}, max={returns.max().item():.4f}")
            print(f"NaN count: {torch.isnan(returns).sum().item()}")
        feature_list.append(returns)
        feature_names.append(f'{period}d_return')

    # Calculate SMA features
    for period in sma_periods:
        sma = calculate_sma(prices, period)
        # Calculate percentage difference from SMA
        sma_diff = (prices - sma) / (sma + 1e-8)  # Add epsilon to prevent division by zero
        if debug:
            print(f"\nSMA {period} diff:")
            print(f"Shape: {sma_diff.shape}")
            print(f"Stats: min={sma_diff.min().item():.4f}, max={sma_diff.max().item():.4f}")
            print(f"NaN count: {torch.isnan(sma_diff).sum().item()}")
        feature_list.append(sma_diff)
        feature_names.append(f'sma{period}_diff')

    if use_volatility:
        # Add realized volatility features
        for period in return_periods:
            returns = calculate_returns(prices, period)
            # Use rolling window for volatility calculation
            volatility = torch.zeros_like(returns)

            # Ensure minimum window size for std calculation
            min_window = max(5, period // 5)  # Use at least 5 points or 1/5 of period

            for i in range(min_window, len(returns)):
                # Use unbiased estimator and ensure minimum window size
                window = returns[max(0, i-period):i]
                if len(window) >= min_window:
                    volatility[i] = window.std(unbiased=True)

            if debug:
                print(f"\n{period}-day volatility:")
                print(f"Shape: {volatility.shape}")
                print(f"Stats: min={volatility.min().item():.4f}, max={volatility.max().item():.4f}")
                print(f"NaN count: {torch.isnan(volatility).sum().item()}")

            feature_list.append(volatility)
            feature_names.append(f'{period}d_volatility')

    if use_momentum:
        # Add momentum indicators
        for period in momentum_periods:
            momentum = calculate_returns(prices, period)
            if debug:
                print(f"\n{period}-day momentum:")
                print(f"Shape: {momentum.shape}")
                print(f"Stats: min={momentum.min().item():.4f}, max={momentum.max().item():.4f}")
                print(f"NaN count: {torch.isnan(momentum).sum().item()}")
            feature_list.append(momentum)
            feature_names.append(f'{period}d_momentum')

    # Stack all features first, then slice
    features = torch.stack(feature_list, dim=1)  # Shape: (n_samples, n_features)

    # Calculate target variables (future returns)
    target_list = []
    target_names = []

    for period in target_periods:
        future_returns = calculate_returns(prices, period)
        if debug:
            print(f"\n{period}-day future returns:")
            print(f"Shape: {future_returns.shape}")
            print(f"Stats: min={future_returns.min().item():.4f}, max={future_returns.max().item():.4f}")
            print(f"NaN count: {torch.isnan(future_returns).sum().item()}")
        target_list.append(future_returns)
        target_names.append(f'{period}d_future_return')

    targets = torch.stack(target_list, dim=1)  # Shape: (n_samples, n_targets)

    # Now slice both features and targets after stacking
    features = features[max_lookback:]
    targets = targets[max_lookback:]

    if debug:
        print("\n=== Final Feature Stats ===")
        print(f"Features shape: {features.shape}")
        print(f"Features stats: min={features.min().item():.4f}, max={features.max().item():.4f}")
        print(f"NaN in features: {torch.isnan(features).sum().item()}")
        print(f"Targets shape: {targets.shape}")
        print(f"Targets stats: min={targets.min().item():.4f}, max={targets.max().item():.4f}")
        print(f"NaN in targets: {torch.isnan(targets).sum().item()}")

    # Create StockFeatures object with valid_start_idx
    return StockFeatures(
        features=features,  # Shape: (n_samples, n_features)
        targets=targets,    # Shape: (n_samples, n_targets)
        feature_names=feature_names,
        target_names=target_names,
        valid_start_idx=max_lookback  # Add this field to track where valid data starts
    )


def augment_price_series(
    price_series: torch.Tensor,
    t_dist_df: int = 5,
    scale_factor: float = 0.1,
    debug: bool = False
) -> torch.Tensor:
    """
    Augment price series by adding noise to daily returns.
    
    Args:
        price_series: Original price series tensor
        t_dist_df: Degrees of freedom for Student-t distribution
        scale_factor: Fraction of standard deviation to use for noise
        debug: Whether to print debug information
    
    Returns:
        Augmented price series tensor
    """
    # Calculate daily returns
    daily_returns = (price_series[1:] - price_series[:-1]) / price_series[:-1]
    
    # Calculate standard deviation of returns
    returns_std = torch.std(daily_returns)
    
    if debug:
        print("\nAugmentation Debug:")
        print(f"Original returns std: {returns_std:.6f}")
        print(f"First few original returns: {daily_returns[:5].tolist()}")
    
    # Generate noise from Student-t distribution
    noise = torch.tensor(
        stats.t.rvs(
            df=t_dist_df,
            size=len(daily_returns),
            random_state=None  # Let it use numpy's current random state
        ),
        dtype=torch.float32
    )
    
    # Scale noise by specified fraction of standard deviation
    scaled_noise = noise * returns_std * scale_factor
    
    if debug:
        print(f"Noise std: {torch.std(scaled_noise):.6f}")
        print(f"First few noise values: {scaled_noise[:5].tolist()}")
    
    # Add noise to returns
    augmented_returns = daily_returns + scaled_noise
    
    # Reconstruct price series
    augmented_prices = torch.zeros_like(price_series)
    augmented_prices[0] = price_series[0]  # Keep first price unchanged
    
    # Reconstruct prices from returns
    for i in range(len(augmented_returns)):
        augmented_prices[i + 1] = augmented_prices[i] * (1 + augmented_returns[i])
    
    if debug:
        print("\nPrice series comparison:")
        print(f"Original first few prices: {price_series[:5].tolist()}")
        print(f"Augmented first few prices: {augmented_prices[:5].tolist()}")
        print(f"Relative difference: {((augmented_prices - price_series) / price_series)[:5].tolist()}")
    
    return augmented_prices

def combine_price_series(
    price_series_list: List[torch.Tensor],
    augmentation_params: Optional[dict] = None,
    debug: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Combine multiple price series and optionally create augmented version.
    
    Args:
        price_series_list: List of price series tensors
        augmentation_params: Optional dict with augmentation parameters
        debug: Whether to print debug information
    
    Returns:
        If augmentation_params is None:
            Combined price series tensor
        If augmentation_params is provided:
            Tuple of (original combined series, augmented combined series)
    """
    # Original combination logic
    combined_series = combine_price_series_base(price_series_list, debug)
    
    # If augmentation is not requested, return original series
    if augmentation_params is None or not augmentation_params.get('enabled', False):
        return combined_series
    
    # Create augmented version
    augmented_series = augment_price_series(
        combined_series,
        t_dist_df=augmentation_params.get('t_dist_df', 5),
        scale_factor=augmentation_params.get('scale_factor', 0.1),
        debug=debug
    )
    
    return combined_series, augmented_series

# Rename original combine_price_series to combine_price_series_base
def combine_price_series_base(price_series_list: List[torch.Tensor], debug: bool = False) -> torch.Tensor:
    """Original combine_price_series implementation"""
    if not price_series_list:
        raise ValueError("Empty price series list")

    if debug:
        print("\nCombining price series:")
        print(f"Number of series: {len(price_series_list)}")
        for i, series in enumerate(price_series_list):
            print(f"\nSeries {i}:")
            print(f"Length: {len(series)}")
            print(f"First few values: {series[:5].tolist()}")
            print(f"Non-zero values: {(series != 0).sum().item()}")
            print(f"Finite values: {torch.isfinite(series).sum().item()}")

    # Find the first valid value for each series
    cleaned_series = []
    for i, series in enumerate(price_series_list):
        # Find first non-zero, finite value
        valid_mask = (series != 0) & torch.isfinite(series)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            if debug:
                print(f"\nWarning: Series {i} has no valid values, skipping")
            continue
            
        first_valid_idx = valid_indices[0]
        if debug:
            print(f"\nSeries {i} starts at index {first_valid_idx}")
            
        # Only take data from first valid value onwards
        valid_series = series[first_valid_idx:]
        cleaned_series.append(valid_series)

    if not cleaned_series:
        raise ValueError("No valid price series after cleaning")

    # Add small epsilon to prevent division by zero
    eps = 1e-8
    normalized_series = [cleaned_series[0]]

    for i in range(1, len(cleaned_series)):
        current_series = cleaned_series[i]
        prev_series_end = normalized_series[-1][-1]
        
        # Add epsilon to denominator to prevent division by zero
        scaling_factor = prev_series_end / (current_series[0] + eps)
        
        # Check for invalid scaling factors
        if not torch.isfinite(scaling_factor):
            if debug:
                print(f"\nWarning: Invalid scaling factor detected for series {i}")
                print(f"prev_series_end: {prev_series_end}")
                print(f"current_series[0]: {current_series[0]}")
            scaling_factor = 1.0
            
        normalized_series.append(current_series * scaling_factor)

    combined_series = torch.cat(normalized_series)
    
    if debug:
        print("\nCombined series statistics:")
        print(f"Length: {len(combined_series)}")
        print(f"First few values: {combined_series[:5].tolist()}")
        print(f"Non-zero values: {(combined_series != 0).sum().item()}")
        print(f"Finite values: {torch.isfinite(combined_series).sum().item()}")

    return combined_series

def normalize_features(features: torch.Tensor, eps: float = 1e-8, debug: bool = False) -> torch.Tensor:
    """
    Normalize features using z-score normalization with safeguards for edge cases.

    Args:
        features: Input feature tensor
        eps: Small constant to prevent division by zero
        debug: Whether to print debug information

    Returns:
        Normalized features tensor
    """
    if debug:
        print("\nNormalizing features:")
        print(f"Input shape: {features.shape}")
        print(f"Input stats:")
        print(f"  Mean: {features.mean(dim=0)}")
        print(f"  Std: {features.std(dim=0)}")
        print(f"  Min: {features.min(dim=0)[0]}")
        print(f"  Max: {features.max(dim=0)[0]}")
        
        # Print first few rows of input
        print("\nFirst few input rows:")
        for i in range(min(3, len(features))):
            print(f"Row {i}: {features[i].tolist()}")

    # If we have only 1 sample, return as is
    if features.size(0) <= 1:
        if debug:
            print("Only 1 sample, returning as is")
        return features

    mean = features.mean(dim=0, keepdim=True)

    # Calculate std with unbiased=False to avoid the degrees of freedom warning
    # This is fine for normalization purposes
    std = features.std(dim=0, keepdim=True, unbiased=False)

    if debug:
        print("\nStd calculation diagnostics:")
        print(f"Contains complex numbers: {torch.is_complex(std)}")
        print(f"Contains NaN: {torch.isnan(std).any()}")
        print(f"Contains Inf: {torch.isinf(std).any()}")
        print(f"Min value: {std.min().item()}")
        print(f"Max value: {std.max().item()}")

    # Replace zero/near-zero std values with 1 to avoid division issues
    std = torch.where(std < eps, torch.ones_like(std), std)

    if debug:
        print("\nNormalization parameters:")
        print(f"Mean shape: {mean.shape}")
        print(f"Std shape: {std.shape}")
        print(f"Mean: {mean.squeeze().tolist()}")
        print(f"Std: {std.squeeze().tolist()}")
        print(f"Features with std < eps: {(std < eps).sum().item()}")

    # Normalize with additional checks
    if debug:
        if torch.any(torch.isnan(mean)) or torch.any(torch.isnan(std)):
            print("Warning: NaN values in normalization parameters!")
        if torch.any(torch.isinf(mean)) or torch.any(torch.isinf(std)):
            print("Warning: Inf values in normalization parameters!")

    # Normalize
    normalized = (features - mean) / std

    if debug:
        print("\nOutput stats:")
        print(f"  Mean: {normalized.mean(dim=0)}")
        print(f"  Std: {normalized.std(dim=0)}")
        print(f"  Min: {normalized.min(dim=0)[0]}")
        print(f"  Max: {normalized.max(dim=0)[0]}")
        
        # Print first few rows of output
        print("\nFirst few normalized rows:")
        for i in range(min(3, len(normalized))):
            print(f"Row {i}: {normalized[i].tolist()}")

        # Check for any extreme values
        extreme_mask = torch.abs(normalized) > 5
        if extreme_mask.any():
            print("\nWarning: Found extreme normalized values (|x| > 5):")
            extreme_indices = torch.where(extreme_mask)
            for i, j in zip(*extreme_indices):
                print(f"  Position ({i},{j}): {normalized[i,j].item():.4f}")
                print(f"    Original value: {features[i,j].item():.4f}")
                print(f"    Mean: {mean[0,j].item():.4f}")
                print(f"    Std: {std[0,j].item():.4f}")

    return normalized
