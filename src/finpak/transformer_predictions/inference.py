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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    """
    plt.figure(figsize=(15, 8))
    
    # Print diagnostic about price series
    # print(f"\nPrice series stats:")
    # print(f"Length: {len(original_prices)}")
    # print(f"Non-NaN values: {torch.isfinite(original_prices).sum().item()}")
    # print(f"Value range: [{original_prices[torch.isfinite(original_prices)].min().item():.2f}, {original_prices[torch.isfinite(original_prices)].max().item():.2f}]")
    
    # Find the earliest start index and latest end point
    earliest_start = min(start_indices)
    latest_end = max([start_idx + len(pred) for start_idx, pred in zip(start_indices, predictions_list)])
    
    # Plot only relevant historical prices
    hist_start = max(0, earliest_start - window_size)  # Start window_size points before earliest prediction
    hist_end = latest_end  # End at the last prediction point
    
    # Create arrays for valid historical data points
    hist_prices = original_prices[hist_start:hist_end]
    valid_mask = torch.isfinite(hist_prices)
    valid_indices = torch.arange(hist_start, hist_end)[valid_mask]
    valid_prices = hist_prices[valid_mask]
    
    plt.plot(
        valid_indices.numpy(),
        valid_prices.numpy(),
        label='Historical Prices',
        color='blue',
        linewidth=2
    )
    
    # Plot each prediction sequence
    colors = plt.cm.rainbow(np.linspace(0, 1, len(start_indices)))
    for i, (start_idx, predictions) in enumerate(zip(start_indices, predictions_list)):
        # Ensure start_idx is valid
        if not torch.isfinite(original_prices[start_idx]):
            print(f"\nWarning: Invalid start index {start_idx}, skipping prediction")
            continue
            
        # Print diagnostic information
        # print(f"\nPrediction sequence {i}:")
        # print(f"Start index: {start_idx}")
        # print(f"Number of predictions: {len(predictions)}")
        # print(f"First few predictions (returns): {predictions[:5, 0]}")
        
        # Convert returns to prices
        last_known_price = original_prices[start_idx].item()
        pred_prices = [last_known_price]
        
        for pred_return in predictions[:, 0]:
            next_price = pred_prices[-1] * (1 + pred_return.item())
            pred_prices.append(next_price)
        
        pred_prices = pred_prices[1:]  # Remove the initial price
        
        # Print diagnostic information about prices
        # print(f"Last known price: {last_known_price}")
        # print(f"First few predicted prices: {pred_prices[:5]}")
        
        # Plot predicted prices
        pred_idx = range(start_idx, start_idx + len(pred_prices))
        plt.plot(
            pred_idx,
            pred_prices,
            label=f'Prediction {i} (t={start_idx})',
            color=colors[i],
            linewidth=2,
            linestyle='--'
        )
        
        plt.axvline(x=start_idx, color=colors[i], linestyle=':', alpha=0.3)
    
    plt.title('Stock Price Predictions', fontsize=14)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10)
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
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Validate start indices
    valid_indices = []
    for idx in start_indices:
        if idx >= len(price_series):
            print(f"Warning: Start index {idx} is beyond price series length {len(price_series)}")
            continue
        if not torch.isfinite(price_series[idx]):
            print(f"Warning: Invalid price at index {idx}")
            continue
        valid_indices.append(idx)
    
    if not valid_indices:
        raise ValueError("No valid start indices provided")
    
    # print(f"\nMaking predictions for {len(valid_indices)} start points:")
    # print(f"Valid start indices: {valid_indices}")
    # print(f"Price series length: {len(price_series)}")
    # print(f"Predicting {n_steps} steps for each start point")
    
    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path,
        model_params,
        device=device
    )
    
    # Make predictions for each start index
    predictions_list = []
    for start_idx in valid_indices:
        # print(f"\nProcessing start index {start_idx}")
        
        # Get initial sequence
        initial_sequence = price_series[max(0, start_idx - sequence_length):start_idx + 1]
        # print(f"Initial sequence length: {len(initial_sequence)}")
        
        # Make prediction
        predictions = make_autoregressive_prediction(
            model,
            initial_sequence,
            n_steps,
            device,
            return_periods,
            sma_periods
        )
        # print(f"Generated {len(predictions)} predictions")
        predictions_list.append(predictions)
    
    # Plot results
    plot_predictions(price_series, valid_indices, predictions_list)
