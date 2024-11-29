import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class StockFeatures:
    """Container for processed stock features and targets"""
    continuous_features: torch.Tensor  # Shape: (n_samples, n_continuous_features)
    categorical_features: torch.Tensor  # Shape: (n_samples, n_categorical_features)
    targets: torch.Tensor   # Shape: (n_samples, n_targets)
    continuous_feature_names: list[str]
    categorical_feature_names: list[str]
    target_names: list[str]
    valid_start_idx: int    # Index where features become valid after initialization period

class StockDataset(Dataset):
    def __init__(
        self,
        continuous_features: torch.Tensor,
        categorical_features: torch.Tensor,
        targets: torch.Tensor,
        sequence_length: int = 60,
        continuous_feature_names: Optional[list[str]] = None,
        categorical_feature_names: Optional[list[str]] = None,
        target_names: Optional[list[str]] = None,
        valid_start_idx: int = 0,
        normalize_continuous: bool = True  # Whether to normalize continuous features
    ):
        """
        Dataset for sequence prediction
        
        Args:
            continuous_features: Tensor of shape (n_samples, n_continuous_features)
            categorical_features: Tensor of shape (n_samples, n_categorical_features)
            targets: Tensor of shape (n_samples, n_targets)
            sequence_length: Number of time steps to use as input
            continuous_feature_names: Optional list of continuous feature names
            categorical_feature_names: Optional list of categorical feature names
            target_names: Optional list of target names
            valid_start_idx: Index where features become valid after initialization period
            normalize_continuous: Whether to normalize continuous features using z-score normalization
        """
        if len(continuous_features) != len(targets) or len(categorical_features) != len(targets):
            raise ValueError("Features and targets must have same length")
            
        self.sequence_length = sequence_length
        self.continuous_feature_names = continuous_feature_names
        self.categorical_feature_names = categorical_feature_names
        self.target_names = target_names
        self.valid_start_idx = valid_start_idx
        
        # Normalize continuous features if requested
        if normalize_continuous:
            # Calculate mean and std across all samples
            mean = continuous_features.mean(dim=0, keepdim=True)
            std = continuous_features.std(dim=0, keepdim=True, unbiased=False)
            # Replace zero/near-zero std values with 1
            std = torch.where(std < 1e-6, torch.ones_like(std), std)
            # Normalize
            self.continuous_features = (continuous_features - mean) / std
        else:
            self.continuous_features = continuous_features
            
        # Categorical features are already encoded as integers, no normalization needed
        self.categorical_features = categorical_features
        self.targets = targets
        
    def __len__(self) -> int:
        """Return number of sequences that can be created"""
        # Account for sequence length and valid start index
        return len(self.continuous_features) - self.sequence_length - self.valid_start_idx + 1
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sequence of features and corresponding target
        
        Args:
            idx: Index of sequence
            
        Returns:
            Tuple of (continuous_sequence, categorical_sequence, target)
        """
        # Adjust index for valid start
        idx = idx + self.valid_start_idx
        
        # Get sequence of features
        continuous_sequence = self.continuous_features[idx:idx + self.sequence_length]
        categorical_sequence = self.categorical_features[idx:idx + self.sequence_length]
        
        # Get target (use last value in sequence)
        target = self.targets[idx + self.sequence_length - 1]
        
        # Return three separate tensors
        return continuous_sequence, categorical_sequence, target
