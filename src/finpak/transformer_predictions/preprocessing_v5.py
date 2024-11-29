import torch
from typing import List, Optional
from stock_dataset_v5 import StockFeatures


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
    returns = (prices[period:] - prices[:-period]) / (prices[:-period] + eps)

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

def create_price_change_bins(returns: torch.Tensor, n_bins: int, min_val: Optional[float] = None, max_val: Optional[float] = None) -> torch.Tensor:
    """
    Convert continuous returns into categorical bins
    
    Args:
        returns: Tensor of price returns
        n_bins: Number of bins to create
        min_val: Minimum value for binning. If None, uses data minimum
        max_val: Maximum value for binning. If None, uses data maximum
    
    Returns:
        Tensor of bin indices (0 to n_bins-1)
    """
    if min_val is None:
        min_val = returns.min()
    if max_val is None:
        max_val = returns.max()
    
    # Add small epsilon to max_val to ensure the maximum value falls into the last bin
    eps = 1e-8
    bins = torch.linspace(min_val, max_val + eps, n_bins + 1)
    return torch.bucketize(returns, bins) - 1  # -1 to make 0-based indexing

def create_stock_features(
    prices: torch.Tensor,
    config: dict,
    debug: bool = False
) -> StockFeatures:
    """
    Create feature matrix and target variables from price series
    
    Config parameters:
        - return_periods: List of periods for calculating returns
        - sma_periods: List of windows for calculating SMAs
        - target_periods: List of future periods for prediction targets
        - price_change_bins: Optional dict with keys:
            - n_bins: Number of bins for price changes
            - min_val: Optional minimum value for binning
            - max_val: Optional maximum value for binning
        - use_volatility: Whether to include volatility features
        - use_momentum: Whether to include momentum features
        - momentum_periods: List of periods for momentum features
    
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
        print("Config keys:", config.keys())
        if 'data_params' in config:
            print("Data Params:", config['data_params'])
        else:
            print("Warning: 'data_params' not found in config")
        
    # Initialize lists to store features and their names
    continuous_features = []
    categorical_features = []
    continuous_feature_names = []
    categorical_feature_names = []
    
    # Calculate returns for different periods
    for period in config['return_periods']:
        returns = calculate_returns(prices, period, debug)
        continuous_features.append(returns)
        continuous_feature_names.append(f'return_{period}d')
        
        # If volatility features are enabled, add them
        if config.get('use_volatility', False):
            # Calculate rolling standard deviation of returns
            vol = torch.zeros_like(returns)
            for i in range(period, len(returns)):
                window = returns[i-period:i]
                if len(window) > 1:  # Ensure we have at least 2 points for std
                    vol[i] = window.std(unbiased=True)
                else:
                    vol[i] = 0.0  # Set to 0 if not enough data points
            continuous_features.append(vol)
            continuous_feature_names.append(f'volatility_{period}d')
        
        # If price change binning is enabled, add categorical features
        if 'price_change_bins' in config:
            # Only bin the 1-day return
            if period == 1:
                bin_config = config['price_change_bins']
                binned_returns = create_price_change_bins(
                    returns,
                    n_bins=bin_config['n_bins'],
                    min_val=bin_config.get('min_val', None),
                    max_val=bin_config.get('max_val', None)
                )
                categorical_features.append(binned_returns)
                categorical_feature_names.append(f'return_{period}d_bin')
    
    # Calculate SMAs
    for window in config['sma_periods']:
        sma = calculate_sma(prices, window)
        continuous_features.append(sma)
        continuous_feature_names.append(f'sma_{window}d')
    
    # Add momentum features if enabled
    if config.get('use_momentum', False):
        for period in config.get('momentum_periods', []):
            momentum = torch.zeros_like(prices)
            for i in range(period, len(prices)):
                momentum[i] = (prices[i] - prices[i-period]) / prices[i-period]
            continuous_features.append(momentum)
            continuous_feature_names.append(f'momentum_{period}d')
    
    # Stack features
    continuous_tensor = torch.stack(continuous_features, dim=1) if continuous_features else torch.tensor([])
    categorical_tensor = torch.stack(categorical_features, dim=1) if categorical_features else torch.tensor([])
    
    if debug:
        print(f"\nFeature counts:")
        print(f"Continuous features: {len(continuous_feature_names)}")
        print(f"Categorical features: {len(categorical_feature_names)}")
        print(f"\nContinuous feature names: {continuous_feature_names}")
        print(f"Categorical feature names: {categorical_feature_names}")
    
    # Calculate target variables (future returns)
    target_list = []
    target_names = []
    for period in config['target_periods']:
        future_returns = calculate_returns(prices, period)
        if debug:
            print(f"\n{period}-day future returns:")
            print(f"Shape: {future_returns.shape}")
            print(f"Stats: min={future_returns.min().item():.4f}, max={future_returns.max().item():.4f}")
            print(f"NaN count: {torch.isnan(future_returns).sum().item()}")
        target_list.append(future_returns)
        target_names.append(f'future_return_{period}d')
    
    targets = torch.stack(target_list, dim=1)
    
    # Find valid_start_idx (maximum lookback period)
    lookback_periods = (
        config['return_periods'] +
        config['sma_periods'] +
        (config.get('momentum_periods', []) if config.get('use_momentum', False) else [])
    )
    valid_start_idx = max(
        max(lookback_periods),
        max(config['target_periods'])
    )
    
    return StockFeatures(
        continuous_features=continuous_tensor,
        categorical_features=categorical_tensor,
        targets=targets,
        continuous_feature_names=continuous_feature_names,
        categorical_feature_names=categorical_feature_names,
        target_names=target_names,
        valid_start_idx=valid_start_idx
    )


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

    # Replace zero/near-zero std values with 1 to avoid division issues
    std = torch.where(std < eps, torch.ones_like(std), std)

    if debug:
        print("\nNormalization parameters:")
        print(f"Mean: {mean.squeeze().tolist()}")
        print(f"Std: {std.squeeze().tolist()}")
        print(f"Features with std < eps: {(std < eps).sum().item()}")

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
