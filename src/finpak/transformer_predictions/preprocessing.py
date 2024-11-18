import torch
from typing import List
from stock_dataset import StockFeatures


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

    # Add small epsilon to denominator to prevent division by zero
    eps = 1e-8
    returns = (prices[period:] - prices[:-period]) / (prices[:-period] + eps)

    # Replace infinite values with the maximum float value
    returns = torch.nan_to_num(returns, nan=0.0, posinf=3.4e38, neginf=-3.4e38)

    # Debug: Print some sample returns
    if debug:
        print(f"First few {period}-day returns: {returns[:5].tolist()}")
        finite_returns = returns[torch.isfinite(returns)]
        if len(finite_returns) > 0:
            print(f"Return statistics:")
            print(f"Mean: {finite_returns.mean().item():.4f}")
            print(f"Std: {finite_returns.std().item():.4f}")
            print(f"Min: {finite_returns.min().item():.4f}")
            print(f"Max: {finite_returns.max().item():.4f}")
        else:
            print("Warning: No finite returns found!")

    # Pad to maintain length
    padding = torch.zeros(period, dtype=prices.dtype, device=prices.device)
    return torch.cat([padding, returns])

def calculate_sma(prices: torch.Tensor, window: int) -> torch.Tensor:
    """Calculate Simple Moving Average"""
    sma = torch.zeros_like(prices)
    for i in range(window, len(prices) + 1):
        sma[i-1] = prices[i-window:i].mean()
    return sma

def create_stock_features(
    prices: torch.Tensor,
    return_periods: List[int] = [1, 5],
    sma_periods: List[int] = [20],
    target_periods: List[int] = [1, 5],
    use_volatility: bool = False,
    use_momentum: bool = False,
    momentum_periods: List[int] = [9, 28, 47],
    debug: bool = False
) -> StockFeatures:
    """
    Create feature matrix and target variables from price series
    """
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
        print(f"Return periods: {return_periods}")
        print(f"SMA periods: {sma_periods}")
        print(f"Target periods: {target_periods}")
        if use_volatility:
            print(f"Using volatility features with return periods: {return_periods}")
        if use_momentum:
            print(f"Using momentum features with periods: {momentum_periods}")

    # Ensure we have enough data points
    if len(prices) <= max_lookback:
        raise ValueError(f"Price series length ({len(prices)}) must be greater than maximum lookback period ({max_lookback})")

    feature_list = []
    feature_names = []

    # Calculate return features
    for period in return_periods:
        returns = calculate_returns(prices, period, debug=debug)
        feature_list.append(returns[max_lookback:])  # Only use data after initialization
        feature_names.append(f'{period}d_return')

    # Calculate SMA features
    for period in sma_periods:
        sma = calculate_sma(prices, period)
        # Calculate percentage difference from SMA
        sma_diff = (prices - sma) / sma
        feature_list.append(sma_diff[max_lookback:])  # Only use data after initialization
        feature_names.append(f'sma{period}_diff')

    if use_volatility:
        # Add realized volatility features
        for period in return_periods:
            returns = calculate_returns(prices, period)
            volatility = torch.zeros_like(returns)
            for i in range(period, len(returns)):
                volatility[i] = returns[i-period:i].std()
            feature_list.append(volatility[max_lookback:])  # Only use data after initialization
            feature_names.append(f'{period}d_volatility')

    if use_momentum:
        # Add momentum indicators
        for period in momentum_periods:
            momentum = calculate_returns(prices, period)
            feature_list.append(momentum[max_lookback:])  # Only use data after initialization
            feature_names.append(f'{period}d_momentum')

    # Stack features into a single tensor
    features = torch.stack(feature_list, dim=1)

    # Calculate target variables (future returns)
    target_list = []
    target_names = []
    
    for period in target_periods:
        future_returns = calculate_returns(prices, period)
        target_list.append(future_returns[max_lookback:])  # Only use data after initialization
        target_names.append(f'{period}d_future_return')

    targets = torch.stack(target_list, dim=1)

    if debug:
        print(f"\nFeature matrix shape: {features.shape}")
        print(f"Target matrix shape: {targets.shape}")
        print(f"Feature names: {feature_names}")
        print(f"Target names: {target_names}")
        
        # Print statistics for each feature
        print("\nFeature statistics:")
        for i, name in enumerate(feature_names):
            feature_data = features[:, i]
            print(f"\n{name}:")
            print(f"Mean: {feature_data.mean():.4f}")
            print(f"Std: {feature_data.std():.4f}")
            print(f"Min: {feature_data.min():.4f}")
            print(f"Max: {feature_data.max():.4f}")
            print(f"NaN count: {torch.isnan(feature_data).sum()}")

    return StockFeatures(features, targets, feature_names, target_names)


def combine_price_series(price_series_list: List[torch.Tensor], debug: bool = False) -> torch.Tensor:
    """
    Combine multiple price series by normalizing each subsequent series
    to start where the previous one ended. Only uses data from when each ticker
    actually starts trading (i.e., has non-zero, non-NaN values).
    """
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

def normalize_features(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize features using z-score normalization with safeguards for edge cases.

    Args:
        features: Input feature tensor
        eps: Small constant to prevent division by zero

    Returns:
        Normalized features tensor
    """
    # If we have only 1 sample, return as is
    if features.size(0) <= 1:
        return features

    mean = features.mean(dim=0, keepdim=True)

    # Calculate std with unbiased=False to avoid the degrees of freedom warning
    # This is fine for normalization purposes
    std = features.std(dim=0, keepdim=True, unbiased=False)

    # Replace zero/near-zero std values with 1 to avoid division issues
    std = torch.where(std < eps, torch.ones_like(std), std)

    return (features - mean) / std
