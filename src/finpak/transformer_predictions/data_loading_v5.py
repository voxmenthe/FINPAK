from torch.utils.data import DataLoader
import torch
import pandas as pd
from typing import Tuple, List
from stock_dataset_v5 import StockDataset, StockFeatures
from preprocessing_v5 import create_stock_features, combine_price_series


def collate_fn(batch, debug: bool = False):
    """Custom collate function for batching sequences"""
    if debug:
        print("\n=== Debug: Inside module collate_fn ===")
        print("Batch type:", type(batch))
        print("Batch length:", len(batch))
        if batch:
            print("First item type:", type(batch[0]))
            print("First item length:", len(batch[0]))
    
    # Unpack the batch into separate tensors
    continuous_sequences, categorical_sequences, targets = zip(*batch)
    
    if debug:
        print("\nAfter unpacking:")
        print("Continuous sequences type:", type(continuous_sequences))
        print("Continuous sequences length:", len(continuous_sequences))
        print("First continuous sequence shape:", continuous_sequences[0].shape)
        print("Categorical sequences type:", type(categorical_sequences))
        print("First categorical sequence shape:", categorical_sequences[0].shape)
        print("Targets type:", type(targets))
        print("First target shape:", targets[0].shape)
    
    # Stack features into a single tensor (batch_size, sequence_length, n_features)
    continuous_sequences = torch.stack(continuous_sequences)
    categorical_sequences = torch.stack(categorical_sequences)
    
    # Stack targets into a single tensor (batch_size, n_targets)
    targets = torch.stack(targets)

    if debug:
        print("\nFinal tensor shapes:")
        print("Continuous shape:", continuous_sequences.shape)
        print("Categorical shape:", categorical_sequences.shape)
        print("Targets shape:", targets.shape)
        print("=== End Debug ===\n")
    
    # Return three separate tensors
    return continuous_sequences, categorical_sequences, targets


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
        config=config['data_params'],
        debug=debug
    )

    # Process features for validation data
    val_features = create_stock_features(
        prices=val_prices,
        config=config['data_params'],
        debug=debug
    )

    # Create datasets
    train_dataset = StockDataset(
        continuous_features=train_features.continuous_features,
        categorical_features=train_features.categorical_features,
        targets=train_features.targets,
        sequence_length=config['data_params']['sequence_length'],
        continuous_feature_names=train_features.continuous_feature_names,
        categorical_feature_names=train_features.categorical_feature_names,
        target_names=train_features.target_names,
        valid_start_idx=train_features.valid_start_idx  # Pass initialization period
    )

    val_dataset = StockDataset(
        continuous_features=val_features.continuous_features,
        categorical_features=val_features.categorical_features,
        targets=val_features.targets,
        sequence_length=config['data_params']['sequence_length'],
        continuous_feature_names=val_features.continuous_feature_names,
        categorical_feature_names=val_features.categorical_feature_names,
        target_names=val_features.target_names,
        valid_start_idx=val_features.valid_start_idx
    )

    # Create dataloaders
    if debug:
        print("\n=== Debug: Creating DataLoaders ===")
        print("Train dataset type:", type(train_dataset))
        print("Train dataset length:", len(train_dataset))
        print("First item from train dataset:", train_dataset[0])
        print("Batch size:", config['train_params']['batch_size'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_params']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    if debug:
        print("\n=== Debug: Creating Validation DataLoaders ===")
        print("Validation dataset type:", type(val_dataset))
        print("Validation dataset length:", len(val_dataset))
        print("First item from validation dataset:", val_dataset[0])
        print("Batch size:", config['train_params']['batch_size'])

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train_params']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return (
        train_loader,
        val_loader,
        train_features.continuous_feature_names + train_features.categorical_feature_names,
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
        # Find first non-NaN value
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
            
        train_price_series.append(price_tensor)

    # Process validation price series for subset
    val_price_series = []
    for ticker in val_tickers:
        prices = val_df[ticker]
        # Find first non-NaN value
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

    # Combine price series separately for train and validation
    combined_train_prices = combine_price_series(train_price_series, debug=debug)
    combined_val_prices = combine_price_series(val_price_series, debug=debug)

    if debug:
        print(f"\nSubset Information:")
        print(f"Training tickers: {train_tickers}")
        print(f"Validation tickers: {val_tickers}")
        print(f"Training series length: {len(combined_train_prices)}")
        print(f"Validation series length: {len(combined_val_prices)}")

    # Create dataloaders with separate train/val prices
    train_loader, val_loader, feature_names, target_names = create_dataloaders(
        train_prices=combined_train_prices,
        val_prices=combined_val_prices,
        config=config,
        debug=debug
    )

    if debug:
        print(f"\nSubset Information:")
        print(f"Training tickers: {train_tickers}")
        print(f"Validation tickers: {val_tickers}")
        print(f"Training series length: {len(combined_train_prices)}")
        print(f"Validation series length: {len(combined_val_prices)}")

    return train_loader, val_loader
