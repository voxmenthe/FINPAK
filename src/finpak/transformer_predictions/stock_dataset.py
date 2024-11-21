import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class StockFeatures:
    """Container for processed stock features and targets"""
    features: torch.Tensor  # Shape: (n_samples, n_features)
    targets: torch.Tensor   # Shape: (n_samples, n_targets)
    feature_names: list[str]
    target_names: list[str]
    valid_start_idx: int    # Index where features become valid after initialization period

class StockDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        sequence_length: int = 60,
        feature_names: Optional[list[str]] = None,
        target_names: Optional[list[str]] = None,
        valid_start_idx: int = 0  # Add parameter for valid start index
    ):
        """
        Dataset for sequence prediction
        
        Args:
            features: Tensor of shape (n_samples, n_features)
            targets: Tensor of shape (n_samples, n_targets)
            sequence_length: Number of time steps to use as input
            feature_names: Optional list of feature names
            target_names: Optional list of target names
            valid_start_idx: Index where features become valid after initialization period
        """
        if len(features) != len(targets):
            raise ValueError("Features and targets must have same length")
            
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.feature_names = feature_names
        self.target_names = target_names
        self.valid_start_idx = valid_start_idx
        
        if True:
            print(f"Dataset initialized with:")
            print(f"Features shape: {features.shape}")
            print(f"Targets shape: {targets.shape}")
            print(f"Sequence length: {sequence_length}")
        
    def __len__(self):
        # Only return sequences that start after the initialization period
        return len(self.features) - self.sequence_length - self.valid_start_idx
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Feature sequence of shape (sequence_length, n_features)
            y: Target values of shape (n_targets,)
        """
        # Offset idx by valid_start_idx to skip initialization period
        actual_idx = idx + self.valid_start_idx
        
        # Get sequence of features
        x = self.features[actual_idx:actual_idx + self.sequence_length]  # Shape: (sequence_length, n_features)
        
        # Get target values
        y = self.targets[actual_idx + self.sequence_length - 1]  # Shape: (n_targets,)
        
        return x, y
        
    def collate_fn(self, batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function to ensure proper batching of sequences"""
        # Separate features and targets
        features, targets = zip(*batch)
        
        # Stack features into a single tensor (batch_size, sequence_length, n_features)
        features = torch.stack(features)
        
        # Stack targets into a single tensor (batch_size, n_targets)
        targets = torch.stack(targets)
        
        return features, targets
