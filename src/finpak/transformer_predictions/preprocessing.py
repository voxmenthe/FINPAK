import torch
from typing import List
from stock_dataset import StockFeatures


def calculate_returns(prices: torch.Tensor, periods: int) -> torch.Tensor:
    """Calculate percentage returns over specified periods"""
    returns = (prices[periods:] - prices[:-periods]) / prices[:-periods]
    # Pad to maintain length
    padding = torch.zeros(periods, dtype=prices.dtype, device=prices.device)
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
    target_periods: List[int] = [1, 5]
) -> StockFeatures:
    """
    Create feature matrix and target variables from price series
    """
    feature_list = []
    feature_names = []
    
    # Calculate return features
    for period in return_periods:
        returns = calculate_returns(prices, period)
        feature_list.append(returns)
        feature_names.append(f'{period}d_return')
    
    # Calculate SMA features
    for period in sma_periods:
        sma = calculate_sma(prices, period)
        # Calculate percentage difference from SMA
        sma_diff = (prices - sma) / sma
        feature_list.append(sma_diff)
        feature_names.append(f'sma{period}_diff')
    
    # Stack features
    features = torch.stack(feature_list, dim=1)
    
    # Calculate target returns
    target_list = []
    target_names = []
    
    for period in target_periods:
        future_returns = calculate_returns(prices, period)
        # Shift returns back by period to align with current time
        target_returns = future_returns[:-period]
        # Pad end with zeros to maintain length
        target_returns = torch.cat([target_returns, torch.zeros(period)])
        target_list.append(target_returns)
        target_names.append(f'future_{period}d_return')
    
    targets = torch.stack(target_list, dim=1)
    
    # Determine warmup period (maximum lookback needed for any feature)
    warmup_period = max(
        max(return_periods, default=0),
        max(sma_periods, default=0),
        max(target_periods, default=0)
    )
    
    # Remove warmup period and any rows with NaN or infinite values
    valid_rows = torch.isfinite(features).all(dim=1) & torch.isfinite(targets).all(dim=1)
    valid_rows[:warmup_period] = False  # Force first warmup_period rows to be invalid
    
    features = features[valid_rows]
    targets = targets[valid_rows]
    
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