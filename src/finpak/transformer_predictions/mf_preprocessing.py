import torch
from typing import List
from stock_dataset import StockFeatures
from preprocessing import calculate_returns, calculate_sma


def discretize_targets(
  targets: torch.Tensor,
  bin_edges: torch.Tensor
) -> torch.Tensor:
  """
  Discretize continuous targets into bins.

  Args:
  targets: Continuous target values.
  bin_edges: Edges of the bins used for discretization.

  Returns:
  Discretized targets as integer bin indices.
  """
  # Use torch.bucketize to find the bin indices
  # Make sure targets are within the bin range
  targets_clipped = torch.clamp(targets, bin_edges[0], bin_edges[-1] - 1e-6)
  target_bins = torch.bucketize(targets_clipped, bin_edges) - 1  # Adjust for 0-based indexing
  return target_bins


def create_stock_features(
    prices: torch.Tensor,
    return_periods: List[int] = [1, 5],
    sma_periods: List[int] = [20],
    target_periods: List[int] = [1, 5],
    bin_edges: torch.Tensor = None,
    debug: bool = False
) -> StockFeatures:
    """
    Create feature matrix and discretized target variables from price series.
    """
    if debug:
        print("\nCreating stock features:")
        print(f"Price series length: {len(prices)}")
        print(f"First few prices: {prices[:5].tolist()}")
        print(f"Return periods: {return_periods}")
        print(f"Target periods: {target_periods}")
    
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

    # Discretize targets if bin_edges are provided
    if bin_edges is not None:
        # Assuming targets is a 2D tensor
        discrete_targets_list = []
        for i in range(targets.shape[1]):
            discrete_targets = discretize_targets(targets[:, i], bin_edges)
            discrete_targets_list.append(discrete_targets)
        targets = torch.stack(discrete_targets_list, dim=1)
  
    # Remove warmup period and any rows with NaN or infinite values
    valid_rows = torch.isfinite(features).all(dim=1) & torch.isfinite(targets).all(dim=1)
    warmup_period = max(return_periods + sma_periods + target_periods)
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

    # Ensure targets are integer type for classification
    targets = targets.long()
  
    return StockFeatures(
        features=features,
        targets=targets,
        feature_names=feature_names,
        target_names=target_names
    )