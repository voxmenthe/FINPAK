from torch.utils.data import DataLoader
import torch
from typing import Tuple, List
from stock_dataset import StockDataset
from preprocessing import create_stock_features, combine_price_series

def create_dataloaders(
    train_prices: torch.Tensor,
    val_prices: torch.Tensor,
    batch_size: int = 32,
    sequence_length: int = 0,
    num_workers: int = 4,
    return_periods: List[int] = [1, 5],
    sma_periods: List[int] = [20],
    target_periods: List[int] = [1, 5],
    debug: bool = False
) -> Tuple[DataLoader, DataLoader, List[str], List[str]]:
    """
    Create train and validation dataloaders from separate price data
    
    Args:
        train_prices: Tensor of training set price data
        val_prices: Tensor of validation set price data
        batch_size: Batch size for dataloaders
        sequence_length: Length of sequences for transformer
        num_workers: Number of worker processes for data loading
        return_periods: List of periods for calculating returns
        sma_periods: List of periods for calculating SMAs
        target_periods: List of periods for target returns
        debug: Whether to print debug information
    """
    # Process features for training data
    train_features = create_stock_features(
        prices=train_prices,
        return_periods=return_periods,
        sma_periods=sma_periods,
        target_periods=target_periods,
        debug=debug
    )
    
    # Process features for validation data
    val_features = create_stock_features(
        prices=val_prices,
        return_periods=return_periods,
        sma_periods=sma_periods,
        target_periods=target_periods,
        debug=debug
    )
    
    # Create datasets
    train_dataset = StockDataset(
        features=train_features.features,
        targets=train_features.targets,
        sequence_length=sequence_length,
        feature_names=train_features.feature_names,
        target_names=train_features.target_names
    )
    
    val_dataset = StockDataset(
        features=val_features.features,
        targets=val_features.targets,
        sequence_length=sequence_length,
        feature_names=val_features.feature_names,
        target_names=val_features.target_names
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
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