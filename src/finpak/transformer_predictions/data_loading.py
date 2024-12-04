from torch.utils.data import DataLoader
import torch
import pandas as pd
from typing import Tuple, List
from stock_dataset import StockDataset
from preprocessing import create_stock_features, combine_price_series


def create_dataloaders(
    train_prices: torch.Tensor,
    val_prices: torch.Tensor,
    config: dict,
    num_workers: int = 4,
    debug: bool = False
) -> Tuple[DataLoader, DataLoader, List[str], List[str]]:
    """
    Create train and validation dataloaders from separate price data

    Args:
        train_prices: Tensor of training set price data
        val_prices: Tensor of validation set price data
        config: Configuration dictionary where:
            config['data_params'] contains return_periods, sma_periods, target_periods, use_volatility, use_momentum, momentum_periods
            config['train_params'] contains batch_size, and sequence_length
        num_workers: Number of worker processes for data loading
        debug: Whether to print debug information
    """  

    # Process features for training data
    train_features = create_stock_features(
        prices=train_prices,
        config=config,
        debug=debug
    )

    # Process features for validation data
    val_features = create_stock_features(
        prices=val_prices,
        config=config,
        debug=debug
    )

    # Create datasets
    train_dataset = StockDataset(
        features=train_features.features,
        targets=train_features.targets,
        sequence_length=config['data_params']['sequence_length'],
        feature_names=train_features.feature_names,
        target_names=train_features.target_names,
        valid_start_idx=train_features.valid_start_idx  # Pass initialization period
    )

    val_dataset = StockDataset(
        features=val_features.features,
        targets=val_features.targets,
        sequence_length=config['data_params']['sequence_length'],
        feature_names=val_features.feature_names,
        target_names=val_features.target_names,
        valid_start_idx=val_features.valid_start_idx  # Pass initialization period
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_params']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train_params']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return (
        train_loader,
        val_loader,
        train_features.feature_names,
        train_features.target_names
    )


def create_subset_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_tickers: list,
    val_tickers: list,
    config: dict,
    debug: bool = False
) -> tuple:
    """Create dataloaders for the current subset of tickers."""
    
    # Process training price series for subset
    train_price_series = []
    for ticker in train_tickers:
        prices = train_df[ticker]
        first_valid_idx = prices.first_valid_index()
        if first_valid_idx is None:
            print(f"Warning: Ticker {ticker} has no valid prices")
            continue
        prices = prices.loc[first_valid_idx:]
        price_tensor = torch.tensor(prices.to_numpy(), dtype=torch.float32)
        
        if debug:
            print(f"\nTicker {ticker} price statistics:")
            print(f"Start date: {prices.index[0]}")
            print(f"End date: {prices.index[-1]}")
            print(f"Length: {len(price_tensor)}")
            print(f"First few prices: {price_tensor[:5].tolist()}")
            print(f"Non-zero values: {(price_tensor != 0).sum().item()}")
            print(f"Finite values: {torch.isfinite(price_tensor).sum().item()}")
            
        train_price_series.append(price_tensor)

    # Process validation price series for subset
    val_price_series = []
    for ticker in val_tickers:
        prices = val_df[ticker]
        first_valid_idx = prices.first_valid_index()
        if first_valid_idx is None:
            print(f"Warning: Ticker {ticker} has no valid prices")
            continue
            
        # Only take data from first valid value onwards
        prices = prices.loc[first_valid_idx:]
        
        # Convert to tensor
        price_tensor = torch.tensor(prices.to_numpy(), dtype=torch.float32)
        
        if debug:
            print(f"\nTicker {ticker} price statistics:")
            print(f"Start date: {prices.index[0]}")
            print(f"End date: {prices.index[-1]}")
            print(f"Length: {len(price_tensor)}")
            print(f"First few prices: {price_tensor[:5].tolist()}")
            print(f"Non-zero values: {(price_tensor != 0).sum().item()}")
            print(f"Finite values: {torch.isfinite(price_tensor).sum().item()}")
            
        val_price_series.append(price_tensor)

    if not train_price_series or not val_price_series:
        raise ValueError("No valid price series found for training or validation")

    # Combine price series with augmentation for training data
    augmentation_params = config.get('augmentation_params')
    if augmentation_params and augmentation_params.get('enabled', False):
        print(f"Augmenting training data with {augmentation_params['subset_fraction']} of subset")
        orig_train_prices, aug_train_prices = combine_price_series(
            train_price_series,
            augmentation_params=augmentation_params,
            debug=debug
        )
        
        # Process features for both original and augmented data
        orig_train_features = create_stock_features(orig_train_prices, config, debug)
        aug_train_features = create_stock_features(aug_train_prices, config, debug)
        
        # Calculate sizes
        orig_size = len(orig_train_features.features)
        aug_size = int(len(aug_train_features.features) * augmentation_params['subset_fraction'])
        
        # Calculate valid range for random start index
        max_start_idx = len(aug_train_features.features) - aug_size
        if max_start_idx < 0:
            raise ValueError(
                f"Augmentation size ({aug_size}) is larger than available augmented data "
                f"({len(aug_train_features.features)})"
            )
            
        # Randomly select starting index for augmented data slice
        random_start = torch.randint(0, max_start_idx + 1, (1,)).item()
        
        # Combine features and targets using random slice of augmented data
        train_features = torch.cat([
            orig_train_features.features,
            aug_train_features.features[random_start:random_start + aug_size]
        ], dim=0)
        
        train_targets = torch.cat([
            orig_train_features.targets,
            aug_train_features.targets[random_start:random_start + aug_size]
        ], dim=0)
        
        if debug:
            print(f"\nAugmented Data Selection:")
            print(f"Random start index: {random_start}")
            print(f"Slice range: {random_start} to {random_start + aug_size}")

    else:
        # Original processing without augmentation
        combined_train_prices = combine_price_series(train_price_series, debug=debug)
        train_features_obj = create_stock_features(combined_train_prices, config, debug)
        train_features = train_features_obj.features
        train_targets = train_features_obj.targets

    # Process validation data (no augmentation)
    combined_val_prices = combine_price_series(val_price_series, debug=debug)
    val_features_obj = create_stock_features(combined_val_prices, config, debug)

    # Create datasets
    train_dataset = StockDataset(
        features=train_features,
        targets=train_targets,
        sequence_length=config['data_params']['sequence_length'],
        feature_names=val_features_obj.feature_names,  # Use same names as validation
        target_names=val_features_obj.target_names,
        valid_start_idx=val_features_obj.valid_start_idx
    )

    val_dataset = StockDataset(
        features=val_features_obj.features,
        targets=val_features_obj.targets,
        sequence_length=config['data_params']['sequence_length'],
        feature_names=val_features_obj.feature_names,
        target_names=val_features_obj.target_names,
        valid_start_idx=val_features_obj.valid_start_idx
    )

    # Create and return dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_params']['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train_params']['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    return train_loader, val_loader
