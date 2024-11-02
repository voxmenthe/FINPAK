from torch.utils.data import DataLoader, random_split
import torch
from typing import Tuple, List
from stock_dataset import StockDataset
from preprocessing import create_stock_features

def create_dataloaders(
    prices: torch.Tensor,
    batch_size: int = 32,
    sequence_length: int = 60,
    train_ratio: float = 0.8,
    num_workers: int = 4,
    return_periods: List[int] = [1, 5],
    sma_periods: List[int] = [20],
    target_periods: List[int] = [1, 5],
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, List[str], List[str]]:
    """
    Create train and validation dataloaders from price data
    """
    # Process features
    stock_features = create_stock_features(
        prices=prices,
        return_periods=return_periods,
        sma_periods=sma_periods,
        target_periods=target_periods
    )
    
    # Create full dataset
    dataset = StockDataset(
        features=stock_features.features,
        targets=stock_features.targets,
        sequence_length=sequence_length,
        feature_names=stock_features.feature_names,
        target_names=stock_features.target_names
    )
    
    # Calculate split sizes
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    # Create train/validation split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=generator
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
        stock_features.feature_names,
        stock_features.target_names
    )