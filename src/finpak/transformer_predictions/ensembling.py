import torch
from typing import List, Dict
from pathlib import Path
from .timeseries_decoder_v3 import TimeSeriesDecoder

def load_model_checkpoint(checkpoint_path: str, model: TimeSeriesDecoder) -> None:
    """Load a model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: TimeSeriesDecoder instance to load weights into
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in state_dict:
        # If the checkpoint contains model weights in a 'model_state_dict' key
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)

def average_checkpoints(checkpoint_paths: List[str], model_config: Dict) -> TimeSeriesDecoder:
    """Average the weights of multiple model checkpoints.
    
    Args:
        checkpoint_paths: List of paths to model checkpoints
        model_config: Dictionary containing model configuration parameters
        
    Returns:
        TimeSeriesDecoder with averaged weights
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided")
    
    # Create a new model instance to hold the averaged weights
    averaged_model = TimeSeriesDecoder(**model_config)
    
    # Initialize a dictionary to store the sum of weights
    averaged_state = {}
    
    # Load and sum all model weights
    for i, checkpoint_path in enumerate(checkpoint_paths):
        # Create a temporary model to load the checkpoint
        temp_model = TimeSeriesDecoder(**model_config)
        load_model_checkpoint(checkpoint_path, temp_model)
        
        with torch.no_grad():
            if i == 0:
                # For the first model, initialize the sum with its parameters
                for name, param in temp_model.state_dict().items():
                    averaged_state[name] = param.clone()
            else:
                # Add the parameters to the running sum
                for name, param in temp_model.state_dict().items():
                    averaged_state[name] += param
    
    # Compute the average
    num_models = len(checkpoint_paths)
    for name in averaged_state:
        averaged_state[name] = averaged_state[name] / num_models
    
    # Load the averaged weights into the model
    averaged_model.load_state_dict(averaged_state)
    return averaged_model

def save_averaged_model(averaged_model: TimeSeriesDecoder, output_path: str) -> None:
    """Save the averaged model to disk.
    
    Args:
        averaged_model: TimeSeriesDecoder with averaged weights
        output_path: Path where to save the averaged model
    """
    torch.save(averaged_model.state_dict(), output_path)
