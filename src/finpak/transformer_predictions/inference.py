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
    sma_periods: List[int] = [20],
    use_multi_horizon: bool = False,
    horizon_weights: Optional[List[float]] = None,
    debug: bool = False
) -> torch.Tensor:
    """
    Make autoregressive predictions starting from an initial sequence.
    """
    model.eval()
    predictions = []
    
    if use_multi_horizon and horizon_weights is None:
        horizon_weights = [1.0 / len(return_periods)] * len(return_periods)
    
    if debug:
        print(f"\nInitial price: {initial_sequence[-1].item():.2f}")
        print(f"Horizon weights: {horizon_weights}")
    
    # Create initial features
    current_sequence = initial_sequence.clone()
    sequence_length = model.pos_embedding.shape[1]
    
    with torch.no_grad():
        for step in range(n_steps):
            # Create features for current sequence
            features = create_stock_features(
                current_sequence,
                return_periods=return_periods,
                sma_periods=sma_periods,
                target_periods=return_periods,
                debug=debug
            ).features
            
            # Normalize features
            features = normalize_features(features)
            
            # Ensure we have enough data points
            if len(features) < sequence_length:
                pad_length = sequence_length - len(features)
                features = torch.cat([torch.zeros(pad_length, features.shape[1]), features])
            
            # Get last sequence_length points
            input_sequence = features[-sequence_length:].unsqueeze(0).to(device)
            
            # Make prediction
            pred = model(input_sequence)
            raw_pred = pred[0].cpu().numpy()  # Store raw prediction for debugging
            predictions.append(pred[0])
            
            # Update price sequence based on prediction
            last_price = current_sequence[-1].item()
            
            if use_multi_horizon:
                # Simple approach: Just use the 1-day prediction and scale down the N-day prediction
                one_day_return = pred[0][0].item()
                n_day_return = pred[0][1].item()  # Assuming second prediction is N-day return
                scaled_n_day = n_day_return / return_periods[1]  # Simple linear scaling
                
                if debug:  # Print for all steps when debugging
                    print(f"\nStep {step} diagnostics:")
                    print(f"Raw model predictions: {raw_pred}")
                    print(f"1-day predicted return: {one_day_return:.4f}")
                    print(f"{return_periods[1]}-day predicted return: {n_day_return:.4f}")
                    print(f"Scaled {return_periods[1]}-day return (daily): {scaled_n_day:.4f}")
                
                # Simple weighted average of 1-day and scaled N-day predictions
                next_return = (horizon_weights[0] * one_day_return + 
                             horizon_weights[1] * scaled_n_day)
                
                if debug:
                    print(f"Weights: {horizon_weights[0]:.2f} * {one_day_return:.4f} + {horizon_weights[1]:.2f} * {scaled_n_day:.4f}")
                    print(f"Final weighted return: {next_return:.4f}")
                    print(f"Current price: {last_price:.2f}")
                    next_price_debug = last_price * (1 + next_return)
                    print(f"Next price: {next_price_debug:.2f}")
                    print(f"Price change: {(next_price_debug - last_price):.2f}")
                    print(f"Percent change: {(next_return * 100):.2f}%")
            else:
                next_return = pred[0][0].item()
            
            # Apply return to get next price
            next_price = last_price * (1 + next_return)
            
            # Add sanity check for price changes
            if debug and abs(next_return) > 0.2:  # Flag large price changes
                print(f"WARNING: Large price change detected at step {step}!")
                print(f"Return: {next_return:.4f}")
                print(f"Price change: {next_price - last_price:.2f}")
            
            current_sequence = torch.cat([current_sequence, torch.tensor([next_price])])
    
    if debug:
        print(f"\nFinal price: {current_sequence[-1].item():.2f}")
        print(f"Total price change: {(current_sequence[-1].item() - initial_sequence[-1].item()):.2f}")
        print(f"Total percent change: {((current_sequence[-1].item() / initial_sequence[-1].item() - 1) * 100):.2f}%")
    
    return torch.stack(predictions)


def plot_predictions(
    original_prices: torch.Tensor,
    start_indices: List[int],
    predictions_list: List[torch.Tensor],
    window_size: int = 100,
    use_multi_horizon: bool = False,
    horizon_weights: Optional[List[float]] = None,
    return_periods: List[int] = [1, 5]
) -> None:
    """
    Plot the original price series and multiple predicted future price sequences.
    """
    plt.figure(figsize=(15, 8))
    
    # Find the earliest start index and latest end point
    earliest_start = min(start_indices)
    latest_end = max([start_idx + len(pred) for start_idx, pred in zip(start_indices, predictions_list)])
    
    # Plot only relevant historical prices
    hist_start = max(0, earliest_start - window_size)
    hist_end = min(latest_end, len(original_prices))
    
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
        if not torch.isfinite(original_prices[start_idx]):
            print(f"\nWarning: Invalid start index {start_idx}, skipping prediction")
            continue
            
        # Convert returns to prices
        last_known_price = original_prices[start_idx].item()
        pred_prices = [last_known_price]
        
        for pred in predictions:
            if use_multi_horizon:
                # Calculate the weighted return the same way as in make_autoregressive_prediction
                one_day_return = pred[0].item()
                n_day_return = pred[1].item()
                scaled_n_day = n_day_return / return_periods[1]
                next_return = (horizon_weights[0] * one_day_return + 
                             horizon_weights[1] * scaled_n_day)
            else:
                next_return = pred[0].item()
                
            next_price = pred_prices[-1] * (1 + next_return)
            pred_prices.append(next_price)
        
        pred_prices = pred_prices[1:]  # Remove the initial price
        
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
    device: Optional[torch.device] = None,
    use_multi_horizon: bool = False,
    horizon_weights: Optional[List[float]] = None,
    debug: bool = False
) -> None:
    """
    Load a model from checkpoint and make/plot predictions from multiple start points.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_multi_horizon and horizon_weights is None:
        horizon_weights = [1.0 / len(return_periods)] * len(return_periods)
    
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
            sma_periods,
            use_multi_horizon=use_multi_horizon,
            horizon_weights=horizon_weights,
            debug=debug
        )
        # print(f"Generated {len(predictions)} predictions")
        predictions_list.append(predictions)
    
    # Plot results with multi-horizon parameters
    plot_predictions(
        price_series, 
        valid_indices, 
        predictions_list,
        use_multi_horizon=use_multi_horizon,
        horizon_weights=horizon_weights,
        return_periods=return_periods
    )
