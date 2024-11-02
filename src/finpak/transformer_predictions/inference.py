import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path

from timeseries_decoder import TimeSeriesDecoder
from preprocessing import create_stock_features, normalize_features


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_params: dict,
    device: Optional[torch.device] = None
) -> TimeSeriesDecoder:
    """
    Load a trained model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_params: Dictionary of model parameters
        device: Device to load the model to (defaults to available device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with same architecture
    model = TimeSeriesDecoder(
        **model_params,
        dropout=0.0  # Set to 0 for inference
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def make_autoregressive_prediction(
    model: TimeSeriesDecoder,
    initial_sequence: torch.Tensor,
    n_steps: int,
    device: torch.device,
    return_periods: List[int] = [1, 5],
    sma_periods: List[int] = [20]
) -> torch.Tensor:
    """
    Make autoregressive predictions starting from an initial sequence.
    
    Args:
        model: Trained TimeSeriesDecoder model
        initial_sequence: Initial price sequence tensor
        n_steps: Number of steps to predict into the future
        device: Device to run inference on
        return_periods: List of return periods for feature calculation
        sma_periods: List of SMA periods for feature calculation
    
    Returns:
        Tensor of predicted returns for each step
    """
    model.eval()
    predictions = []
    
    # Create initial features
    current_sequence = initial_sequence.clone()
    sequence_length = model.pos_embedding.shape[1]
    
    with torch.no_grad():
        for _ in range(n_steps):
            # Create features for current sequence
            features = create_stock_features(
                current_sequence,
                return_periods=return_periods,
                sma_periods=sma_periods,
                target_periods=[1]  # Only need next step prediction
            ).features
            
            # Normalize features - unsure if necessary
            features = normalize_features(features)
            
            # Ensure we have enough data points
            if len(features) < sequence_length:
                pad_length = sequence_length - len(features)
                # Pad with zeros at the beginning
                features = torch.cat([torch.zeros(pad_length, features.shape[1]), features])
            
            # Get last sequence_length points
            input_sequence = features[-sequence_length:].unsqueeze(0).to(device)
            
            # Make prediction
            pred = model(input_sequence)
            predictions.append(pred[0].cpu())
            
            # Update price sequence with predicted return
            last_price = current_sequence[-1]
            next_price = last_price * (1 + pred[0][0].item())
            current_sequence = torch.cat([current_sequence, torch.tensor([next_price])])
    
    return torch.stack(predictions)


def plot_predictions(
    original_prices: torch.Tensor,
    start_indices: List[int],
    predictions_list: List[torch.Tensor],
    window_size: int = 100
) -> None:
    """
    Plot the original price series and multiple predicted future price sequences.
    
    Args:
        original_prices: Complete original price series
        start_indices: List of indices where predictions start
        predictions_list: List of prediction tensors for each start index
        window_size: Number of historical points to show before first prediction
    """
    # Find the earliest start index and latest end point
    earliest_start = min(start_indices)
    latest_end = max([start_idx + len(pred) for start_idx, pred in zip(start_indices, predictions_list)])
    
    plt.figure(figsize=(15, 8))
    
    # Plot historical prices from earliest start through to end
    hist_idx = range(earliest_start - window_size, latest_end)
    plt.plot(
        hist_idx,
        original_prices[earliest_start - window_size:latest_end],
        label='Historical Prices',
        color='blue',
        linewidth=2
    )
    
    # Plot each prediction sequence
    colors = plt.cm.rainbow(np.linspace(0, 1, len(start_indices)))
    for (start_idx, predictions), color in zip(zip(start_indices, predictions_list), colors):
        # Convert returns to prices starting from the last known price
        last_known_price = original_prices[start_idx].item()
        pred_prices = [last_known_price]
        
        # Calculate subsequent predicted prices using cumulative returns
        for pred_return in predictions[:, 0]:  # Use first return period predictions
            next_price = pred_prices[-1] * (1 + pred_return.item())
            pred_prices.append(next_price)
        
        # Remove the initial price we used for starting point
        pred_prices = pred_prices[1:]
        
        # Create time indices for prediction
        pred_idx = range(start_idx, start_idx + len(predictions))
        
        # Plot predicted prices
        plt.plot(
            pred_idx,
            pred_prices,
            label=f'Prediction from t={start_idx}',
            color=color,
            linewidth=2,
            linestyle='--'
        )
        
        # Add vertical line at prediction start
        plt.axvline(x=start_idx, color=color, linestyle=':', alpha=0.3)
    
    plt.title('Stock Price Predictions', fontsize=14)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def predict_from_checkpoint(
    checkpoint_path: str,
    price_series: torch.Tensor,
    start_indices: List[int],
    n_steps: int,
    model_params: dict,
    sequence_length: int = 47,
    return_periods: List[int] = [1, 5],
    sma_periods: List[int] = [20],
    device: Optional[torch.device] = None
) -> None:
    """
    Load a model from checkpoint and make/plot predictions from multiple start points.
    
    Args:
        checkpoint_path: Path to model checkpoint
        price_series: Complete price series tensor
        start_indices: List of indices to start predictions from
        n_steps: Number of steps to predict for each start point
        sequence_length: Length of input sequence
        return_periods: List of return periods for features
        sma_periods: List of SMA periods for features
        device: Device to run inference on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path,
        model_params,
        device=device
    )
    
    # Make predictions for each start index
    predictions_list = []
    for start_idx in start_indices:
        # Get initial sequence
        initial_sequence = price_series[max(0, start_idx - sequence_length):start_idx + 1]
        
        # Make prediction
        predictions = make_autoregressive_prediction(
            model,
            initial_sequence,
            n_steps,
            device,
            return_periods,
            sma_periods
        )
        predictions_list.append(predictions)
    
    # Plot results
    plot_predictions(price_series, start_indices, predictions_list)
