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

    returns = (prices[period:] - prices[:-period]) / prices[:-period]

    # Debug: Print some sample returns
    if debug:
        print(f"First few {period}-day returns: {returns[:5].tolist()}")
        print(f"Return statistics:")
        print(f"Mean: {returns[torch.isfinite(returns)].mean().item():.4f}")
        print(f"Std: {returns[torch.isfinite(returns)].std().item():.4f}")
        print(f"Min: {returns[torch.isfinite(returns)].min().item():.4f}")
        print(f"Max: {returns[torch.isfinite(returns)].max().item():.4f}")

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
    if debug:
        print("\nCreating stock features:")
        print(f"Price series length: {len(prices)}")
        print(f"First few prices: {prices[:5].tolist()}")
        print(f"Return periods: {return_periods}")
        print(f"SMA periods: {sma_periods}")
        print(f"Target periods: {target_periods}")
        if use_volatility:
            print(f"Using volatility features with return periods: {return_periods}")
        if use_momentum:
            print(f"Using momentum features with periods: {momentum_periods}")

    feature_list = []
    feature_names = []

    # Calculate return features
    for period in return_periods:
        returns = calculate_returns(prices, period, debug=debug)
        feature_list.append(returns)
        feature_names.append(f'{period}d_return')

    # Calculate SMA features
    for period in sma_periods:
        sma = calculate_sma(prices, period)
        # Calculate percentage difference from SMA
        sma_diff = (prices - sma) / sma
        feature_list.append(sma_diff)
        feature_names.append(f'sma{period}_diff')

    if use_volatility:
        # Add realized volatility features
        for period in return_periods:
            returns = calculate_returns(prices, period)
            volatility = torch.zeros_like(returns)
            for i in range(period, len(returns)):
                volatility[i] = returns[i-period:i].std()
            feature_list.append(volatility)
            feature_names.append(f'{period}d_volatility')

    if use_momentum:
        # Add momentum indicators
        for period in momentum_periods:  # Common momentum periods
            momentum = calculate_returns(prices, period)
            # Smooth momentum to reduce noise
            momentum = calculate_sma(momentum, window=5)
            feature_list.append(momentum)
            feature_names.append(f'{period}d_momentum')

    # Stack features
    features = torch.stack(feature_list, dim=1)

    # Calculate target returns
    target_list = []
    target_names = []

    if debug:
        print("\nCalculating target returns:")
    for period in target_periods:
        returns = calculate_returns(prices, period, debug=debug)
        # Shift returns back by period to align with current time
        target_returns = returns[:-period]
        # Pad end with zeros to maintain length
        target_returns = torch.cat([target_returns, torch.zeros(period)])
        valid_returns = target_returns[torch.isfinite(target_returns)]

        if debug:
            print(f"\n{period}-day target returns statistics:")
            print(f"Mean: {valid_returns.mean().item():.4f}")
            print(f"Std: {valid_returns.std().item():.4f}")
            print(f"Min: {valid_returns.min().item():.4f}")
            print(f"Max: {valid_returns.max().item():.4f}")

        target_list.append(target_returns)
        target_names.append(f'future_{period}d_return')

    targets = torch.stack(target_list, dim=1)

    # Determine warmup period
    warmup_period = max(
        max(return_periods, default=0),
        max(sma_periods, default=0),
        max(target_periods, default=0)
    )

    # Remove warmup period and any rows with NaN or infinite values
    valid_rows = torch.isfinite(features).all(dim=1) & torch.isfinite(targets).all(dim=1)
    valid_rows[:warmup_period] = False

    features = features[valid_rows]
    targets = targets[valid_rows]

    if debug:
        print("\nFinal feature/target statistics:")
        print(f"Number of valid samples: {len(features)}")
        print("Feature means:", features.mean(dim=0).tolist())
        print("Feature stds:", features.std(dim=0).tolist())
        print("Target means:", targets.mean(dim=0).tolist())
        print("Target stds:", targets.std(dim=0).tolist())

    return StockFeatures(
        features=features,
        targets=targets,
        feature_names=feature_names,
        target_names=target_names
    )


def combine_price_series(price_series_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Combine multiple price series by normalizing each subsequent series
    to start where the previous one ended.
    """
    if not price_series_list:
        raise ValueError("Empty price series list")

    normalized_series = [price_series_list[0]]

    for i in range(1, len(price_series_list)):
        current_series = price_series_list[i]
        prev_series_end = normalized_series[-1][-1]
        scaling_factor = prev_series_end / current_series[0]
        normalized_series.append(current_series * scaling_factor)

    return torch.cat(normalized_series)

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
