import mlx.core as mx
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StockFeaturesMLX:
    """Container for stock features and metadata"""
    features: mx.array
    targets: mx.array
    feature_names: List[str]
    target_names: List[str]
    valid_start_idx: int


def calculate_returns(prices: np.ndarray, period: int, debug: bool = False) -> np.ndarray:
    """Calculate percentage returns over specified periods"""
    if not isinstance(period, int):
        raise TypeError(f"period must be an integer, got {type(period)} instead")

    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    # Ensure prices is 2D if it's not already
    if len(prices.shape) == 1:
        prices = prices.reshape(-1, 1)
    
    # Add small epsilon to denominator to prevent division by zero
    eps = 1e-8
    denominator = prices[:-period]
    # Handle negative prices by using absolute value for denominator
    # but preserving the sign in the final calculation
    denominator_sign = np.sign(denominator)
    denominator_abs = np.abs(denominator) + eps
    numerator = prices[period:] - prices[:-period]
    returns = numerator / denominator_abs
    # Preserve the original sign
    returns = returns * denominator_sign

    # Replace infinite values with the maximum float value
    returns = np.nan_to_num(returns, nan=0.0, posinf=3.4e38, neginf=-3.4e38)

    # Create padding with correct shape
    padding = np.zeros((period,) + returns.shape[1:], dtype=prices.dtype)
    padded_returns = np.concatenate([padding, returns], axis=0)
    
    # If input was 1D, make output 1D
    if len(prices.shape) == 2 and prices.shape[1] == 1:
        padded_returns = padded_returns.squeeze(-1)
        
    return padded_returns


def calculate_sma(prices: np.ndarray, window: int) -> np.ndarray:
    """Calculate Simple Moving Average"""
    sma = np.zeros_like(prices)
    for i in range(window, len(prices) + 1):
        sma[i-1] = prices[i-window:i].mean()
    return sma


def calculate_volatility(returns: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling volatility"""
    vol = np.zeros_like(returns)
    for i in range(window, len(returns) + 1):
        vol[i-1] = returns[i-window:i].std()
    return vol


def calculate_momentum(prices: np.ndarray, window: int) -> np.ndarray:
    """Calculate momentum indicator"""
    momentum = np.zeros_like(prices)
    momentum[window:] = prices[window:] / prices[:-window] - 1
    return momentum


def create_stock_features(
    prices: np.ndarray,
    sequence_length: int,
    return_periods: List[int],
    sma_periods: List[int],
    target_periods: List[int],
    use_volatility: bool = False,
    use_momentum: bool = False,
    momentum_periods: Optional[List[int]] = None,
    debug: bool = False
) -> StockFeaturesMLX:
    """
    Create feature matrix and target variables from price series
    """
    features_list = []
    feature_names = []
    
    # Calculate returns for all periods (both features and targets)
    all_periods = sorted(list(set(return_periods + target_periods)))
    returns_dict = {
        period: calculate_returns(prices, period, debug)
        for period in all_periods
    }
    
    # Add return features
    for period in return_periods:
        features_list.append(returns_dict[period])
        feature_names.append(f"return_{period}")
    
    # Add SMA features
    for period in sma_periods:
        sma = calculate_sma(prices, period)
        features_list.append(sma)
        feature_names.append(f"sma_{period}")
    
    # Add volatility features if requested
    if use_volatility:
        for period in return_periods:
            vol = calculate_volatility(returns_dict[period], period)
            features_list.append(vol)
            feature_names.append(f"volatility_{period}")
    
    # Add momentum features if requested
    if use_momentum and momentum_periods:
        for period in momentum_periods:
            mom = calculate_momentum(prices, period)
            features_list.append(mom)
            feature_names.append(f"momentum_{period}")
    
    # Stack all features
    features = np.stack(features_list, axis=-1)
    
    # Create targets
    targets_list = [returns_dict[period] for period in target_periods]
    targets = np.stack(targets_list, axis=-1)
    target_names = [f"target_{period}" for period in target_periods]
    
    # Calculate valid start index (maximum of all lookback periods)
    valid_start_idx = max(
        max(return_periods + sma_periods + (momentum_periods or [])),
        sequence_length
    )
    
    # Create overlapping sequences
    n_samples = len(features) - sequence_length + 1
    feature_sequences = np.zeros((n_samples, sequence_length, features.shape[-1]))
    target_sequences = np.zeros((n_samples, targets.shape[-1]))
    
    for i in range(n_samples):
        feature_sequences[i] = features[i:i + sequence_length]
        target_sequences[i] = targets[i + sequence_length - 1]  # Use target at end of sequence
    
    return StockFeaturesMLX(
        features=mx.array(feature_sequences),
        targets=mx.array(target_sequences),
        feature_names=feature_names,
        target_names=target_names,
        valid_start_idx=valid_start_idx
    )


def combine_price_series(
    df,
    sequence_length: int,
    return_periods: List[int],
    sma_periods: List[int],
    target_periods: List[int],
    use_volatility: bool = False,
    use_momentum: bool = False,
    momentum_periods: Optional[List[int]] = None,
    debug: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Process price series data into features and targets."""
    
    # Convert DataFrame to numpy array, handling each ticker
    all_features = []
    all_targets = []
    
    for ticker in df.columns:
        # Get price series and convert to numpy
        prices = df[ticker].values
        
        # Skip if all prices are NaN
        if np.all(np.isnan(prices)):
            continue
        
        # Forward fill NaN values
        mask = np.isnan(prices)
        idx = np.where(~mask, np.arange(len(mask)), 0)
        np.maximum.accumulate(idx, out=idx)
        prices = prices[idx]
        
        # Create features for this ticker
        stock_features = create_stock_features(
            prices=prices,
            sequence_length=sequence_length,
            return_periods=return_periods,
            sma_periods=sma_periods,
            target_periods=target_periods,
            use_volatility=use_volatility,
            use_momentum=use_momentum,
            momentum_periods=momentum_periods,
            debug=debug
        )
        
        # Only use data after the valid start index
        valid_features = stock_features.features[stock_features.valid_start_idx:]
        valid_targets = stock_features.targets[stock_features.valid_start_idx:]
        
        # Skip if no valid data
        if len(valid_features) == 0:
            continue
        
        all_features.append(valid_features)
        all_targets.append(valid_targets)
    
    # Concatenate all features and targets
    if not all_features:
        raise ValueError("No valid features found in any price series")
    
    features = np.concatenate(all_features, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    return features, targets


def normalize_features(
    train_features: np.ndarray,
    val_features: np.ndarray,
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization with safeguards for edge cases.
    Normalizes validation features using training set statistics.
    """
    # Calculate mean and std from training set
    mean = np.mean(train_features, axis=0, keepdims=True)
    std = np.std(train_features, axis=0, keepdims=True)
    
    # Replace zero std with 1 to avoid division by zero
    std = np.where(std < eps, 1.0, std)
    
    # Normalize both sets using training statistics
    train_normalized = (train_features - mean) / std
    val_normalized = (val_features - mean) / std
    
    # Clip to reasonable range to handle outliers
    clip_value = 10.0
    train_normalized = np.clip(train_normalized, -clip_value, clip_value)
    val_normalized = np.clip(val_normalized, -clip_value, clip_value)
    
    return train_normalized, val_normalized
