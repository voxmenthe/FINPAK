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

class StockDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        sequence_length: int = 60,
        feature_names: Optional[list[str]] = None,
        target_names: Optional[list[str]] = None
    ):
        """
        Dataset for sequence prediction
        
        Args:
            features: Tensor of shape (n_samples, n_features)
            targets: Tensor of shape (n_samples, n_targets)
            sequence_length: Number of time steps to use as input
            feature_names: Optional list of feature names
            target_names: Optional list of target names
        """
        if len(features) != len(targets):
            raise ValueError("Features and targets must have same length")
            
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.feature_names = feature_names
        self.target_names = target_names
        
    def __len__(self):
        return len(self.features) - self.sequence_length
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Feature sequence of shape (sequence_length, n_features)
            y: Target values of shape (n_targets,)
        """
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return x, y
