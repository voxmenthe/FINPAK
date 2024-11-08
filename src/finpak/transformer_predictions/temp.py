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
  # ... [Same as before up to calculating targets] ...
  # Suppose targets have been calculated as continuous values

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

  # Ensure targets are integer type for classification
  targets = targets.long()
  
  return StockFeatures(
      features=features,
      targets=targets,
      feature_names=feature_names,
      target_names=target_names
  )