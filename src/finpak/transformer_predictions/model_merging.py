import torch
import numpy as np
from typing import Optional


class DAREMerger:
    def __init__(self, base_model: torch.nn.Module):
        self.base_model = base_model
    
    def get_magnitude_mask(self, model: torch.nn.Module, threshold: float = 0.9) -> dict:
        """Create magnitude-based mask for parameter protection"""
        masks = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Get absolute values and threshold
                    abs_values = torch.abs(param.data)
                    cutoff = torch.quantile(abs_values, threshold)
                    # Create binary mask for high-magnitude parameters
                    masks[name] = (abs_values >= cutoff).float()
        return masks

    def compute_delta(self, model_a: torch.nn.Module, model_b: torch.nn.Module) -> dict:
        """Compute parameter differences between models"""
        deltas = {}
        with torch.no_grad():
            for (name_a, param_a), (name_b, param_b) in zip(
                model_a.named_parameters(), model_b.named_parameters()
            ):
                if param_a.requires_grad:
                    deltas[name_a] = param_b.data - param_a.data
        return deltas

    def dare_merge(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        density: float = 0.1,
        seed: Optional[int] = None
    ) -> torch.nn.Module:
        """
        Perform DARE (Drops And REscales) merge
        Args:
            model_a: First model
            model_b: Second model
            density: Percentage of parameters to keep (1 - drop rate)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Get magnitude masks from base model
        magnitude_masks = self.get_magnitude_mask(self.base_model)
        
        # Compute parameter deltas
        deltas = self.compute_delta(model_a, model_b)
        
        # Create merged model starting from model_a
        merged_model = type(model_a)(**model_a.config.__dict__)
        merged_model.load_state_dict(model_a.state_dict())

        with torch.no_grad():
            for name, param in merged_model.named_parameters():
                if param.requires_grad:
                    delta = deltas[name]
                    mask = magnitude_masks[name]
                    
                    # Random selection for DARE
                    random_mask = torch.rand_like(delta) < density
                    
                    # Combine masks and rescale
                    final_mask = random_mask & (mask == 0)
                    scale_factor = 1.0 / density if density > 0 else 1.0
                    
                    # Apply masked and scaled updates
                    param.data += delta * final_mask * scale_factor

        return merged_model

    def ties_merge(
        self,
        models: list[torch.nn.Module],
        weights: Optional[list[float]] = None
    ) -> torch.nn.Module:
        """
        Perform TIES (TrIm Elect Sign and merge) merge
        Args:
            models: List of models to merge
            weights: Optional weights for each model
        """
        if weights is None:
            weights = [1.0] * len(models)
        
        # Normalize weights
        weights = torch.tensor(weights) / sum(weights)
        
        # Initialize merged model from first model
        merged_model = type(models[0])(**models[0].config.__dict__)
        merged_model.load_state_dict(models[0].state_dict())

        with torch.no_grad():
            for name, param in merged_model.named_parameters():
                if param.requires_grad:
                    # Collect parameters from all models
                    params = torch.stack([
                        model.state_dict()[name] for model in models
                    ])
                    
                    # Calculate weighted average
                    weighted_params = (params * weights.view(-1, 1, 1))
                    
                    # Get sign agreement
                    signs = torch.sign(weighted_params)
                    sign_agreement = torch.abs(torch.sum(signs, dim=0)) == len(models)
                    
                    # Only merge parameters with matching signs
                    param.data = torch.where(
                        sign_agreement,
                        torch.sum(weighted_params, dim=0),
                        param.data
                    )

        return merged_model


"""
# Initialize models
model_a = YourModelClass(...)
model_b = YourModelClass(...)
base_model = YourModelClass(...)

# Initialize merger
merger = DAREMerger(base_model)

# Perform DARE merge
merged_model = merger.dare_merge(
    model_a=model_a,
    model_b=model_b,
    density=0.1,  # Keep 10% of parameters
    seed=42
)

# Or perform TIES merge with multiple models
models = [model_a, model_b, model_c]
weights = [1.0, 0.5, 0.5]
merged_model = merger.ties_merge(
    models=models,
    weights=weights
)
"""