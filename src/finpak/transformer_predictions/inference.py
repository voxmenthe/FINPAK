import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path

from timeseries_decoder_v3 import TimeSeriesDecoder
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
    use_forcing: bool = False,
    forcing_halflife: float = 3.0,
    true_future: Optional[torch.Tensor] = None,
    debug: bool = False,
    stability_threshold: float = 0.1,  # Max allowed daily return magnitude
    dampening_factor: float = 0.95,    # Exponential dampening per step
    use_ewma_smoothing: bool = True,   # Use exponential moving average
    ewma_alpha: float = 0.7,          # EWMA smoothing factor
    temperature: float = 0.01,        # Temperature for return sampling (default 0.01 = 1% std)
    use_sampling: bool = False        # Whether to use stochastic sampling
) -> torch.Tensor:
    """
    Make autoregressive predictions starting from an initial sequence.
    
    Args:
        use_forcing: Whether to use exponentially decaying teacher forcing
        forcing_halflife: Number of steps over which forcing influence reduces by half
        true_future: Optional tensor of true future values for forcing (must be at least n_steps long)
    """
    model.eval()
    predictions = []
    
    if use_forcing and true_future is None:
        raise ValueError("true_future must be provided when use_forcing is True")
    
    if use_multi_horizon and horizon_weights is None:
        horizon_weights = [1.0 / len(return_periods)] * len(return_periods)
    
    # Calculate forcing decay factor
    if use_forcing:
        decay_factor = 0.5 ** (1.0 / forcing_halflife)
        forcing_weight = 1.0
        
        if debug:
            print(f"\nUsing teacher forcing with halflife {forcing_halflife}")
            print(f"Decay factor per step: {decay_factor:.4f}")
    
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
            
            # Check if we have enough valid features after warmup
            if len(features) < 1:
                raise ValueError(f"No valid features after warmup period. Initial sequence length: {len(current_sequence)}")
            
            # Ensure we have enough data points for the sequence
            if len(features) < sequence_length:
                pad_length = sequence_length - len(features)
                if debug:
                    print(f"Padding {pad_length} timesteps with zeros")
                features = torch.cat([torch.zeros(pad_length, features.shape[1]), features])
            
            # Get last sequence_length points
            input_sequence = features[-sequence_length:].unsqueeze(0).to(device)
            
            # Make prediction
            pred = model(input_sequence)
            
            # Apply temperature-based sampling if enabled
            if use_sampling:
                # Create a Gaussian distribution centered on predicted returns
                # Temperature is interpreted as the standard deviation in return space
                # e.g., temperature=0.01 means returns vary by ±1% (one std)
                dist = torch.distributions.Normal(pred, temperature)
                
                # Sample from the distribution
                pred = dist.sample()
                
                # Ensure sampled returns are within reasonable bounds
                # Clip to ±3 standard deviations
                pred = torch.clamp(pred, pred - 3*temperature, pred + 3*temperature)
                
                if debug:
                    print(f"\nStep {step} stochastic sampling:")
                    print(f"Original returns: {pred[0].cpu().numpy()}")
                    print(f"Temperature (return std): {temperature:.3f}")
                    print(f"Sampled returns: {pred[0].cpu().numpy()}")
            
            raw_pred = pred[0].cpu().numpy()
            predictions.append(pred[0])
            
            # Update price sequence based on prediction
            last_price = current_sequence[-1].item()
            
            if use_multi_horizon:
                # Get predictions for each horizon
                horizon_returns = []
                for i, period in enumerate(return_periods):
                    # Scale down the N-day return to a daily return
                    period_return = pred[0][i].item()
                    daily_return = (1 + period_return) ** (1/period) - 1
                    
                    # Apply dampening based on horizon length
                    # Longer horizons get more dampening
                    horizon_dampening = dampening_factor ** (period - 1)
                    damped_return = daily_return * horizon_dampening
                    
                    horizon_returns.append(damped_return)
                    
                    if debug and step < 2:  # Show first few steps
                        print(f"\nStep {step} - {period}-day horizon:")
                        print(f"Raw {period}-day return: {period_return:.4f}")
                        print(f"Scaled daily return: {daily_return:.4f}")
                        print(f"Dampened return: {damped_return:.4f}")
                
                # Combine predictions using horizon weights
                next_return = sum(w * r for w, r in zip(horizon_weights, horizon_returns))
                
                # Apply stability controls
                if abs(next_return) > stability_threshold:
                    # Clip extreme predictions while preserving direction
                    next_return = stability_threshold * (next_return / abs(next_return))
                    if debug:
                        print(f"Return exceeded threshold, clipped to: {next_return:.4f}")
                
                # Apply exponential moving average smoothing if enabled
                if use_ewma_smoothing and len(predictions) > 0:
                    prev_return = (current_sequence[-1] / current_sequence[-2] - 1).item()
                    smoothed_return = ewma_alpha * next_return + (1 - ewma_alpha) * prev_return
                    next_return = smoothed_return
                
                if debug and step < 2:
                    print(f"\nStep {step} combined prediction:")
                    for period, ret, weight in zip(return_periods, horizon_returns, horizon_weights):
                        print(f"{period}-day contribution: {weight:.2f} * {ret:.4f} = {weight * ret:.4f}")
                    print(f"Final weighted daily return: {next_return:.4f}")
            else:
                next_return = pred[0][0].item()
            
            # Calculate next price
            next_price = last_price * (1 + next_return)
            
            # Apply teacher forcing if enabled
            if use_forcing:
                true_next_price = true_future[step].item()
                forced_price = (forcing_weight * true_next_price + 
                              (1 - forcing_weight) * next_price)
                
                if debug and step < 5:  # Show first few steps of forcing
                    print(f"\nStep {step} forcing:")
                    print(f"Model price: {next_price:.2f}")
                    print(f"True price: {true_next_price:.2f}")
                    print(f"Forcing weight: {forcing_weight:.4f}")
                    print(f"Forced price: {forced_price:.2f}")
                
                next_price = forced_price
                forcing_weight *= decay_factor
            
            # Add sanity check for price changes
            if debug and abs(next_return) > 0.2:
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
            color='red', #colors[i],
            linewidth=2,
            # linestyle='--'
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
    use_forcing: bool = False,
    forcing_halflife: float = 3.0,
    debug: bool = False,
    stability_threshold: float = 0.1,  # Max allowed daily return magnitude
    dampening_factor: float = 0.95,    # Exponential dampening per step
    use_ewma_smoothing: bool = True,   # Use exponential moving average
    ewma_alpha: float = 0.7,          # EWMA smoothing factor
    temperature: float = 0.01,        # Temperature for return sampling (default 0.01 = 1% std)
    use_sampling: bool = False        # Whether to use stochastic sampling
) -> None:
    """
    Load a model from checkpoint and make/plot predictions from multiple start points.
    
    Args:
        use_forcing: Whether to use exponentially decaying teacher forcing
        forcing_halflife: Number of steps over which forcing influence reduces by half
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_multi_horizon and horizon_weights is None:
        horizon_weights = [1.0 / len(return_periods)] * len(return_periods)
    
    if debug:
        print(f"\nValidating start indices:")
        print(f"Price series length: {len(price_series)}")
        print(f"Number of start indices: {len(start_indices)}")
        print(f"Min start index: {min(start_indices)}")
        print(f"Max start index: {max(start_indices)}")
        print(f"Indices exceeding length: {[idx for idx in start_indices if idx >= len(price_series)]}")
    
    # Calculate required warmup period first
    max_lookback = max([
        max(sma_periods) if sma_periods else 0,
        max(return_periods),  # For return features
        max(return_periods)   # For target calculation
    ])
    
    total_required_history = sequence_length + max_lookback
    if debug:
        print(f"\nRequired history:")
        print(f"Sequence length: {sequence_length}")
        print(f"Max lookback: {max_lookback}")
        print(f"Total required history: {total_required_history}")
    
    # Clip indices to valid range and ensure enough history
    valid_indices = []
    skipped_early = 0  # Count indices skipped due to insufficient history
    skipped_invalid = 0  # Count indices skipped due to invalid prices
    
    for idx in start_indices:
        # Clip to valid range
        idx = min(idx, len(price_series) - 1)
        
        # Check if we have enough history
        if idx < total_required_history:
            skipped_early += 1
            continue
            
        # Check for valid price
        if not torch.isfinite(price_series[idx]):
            skipped_invalid += 1
            continue
            
        valid_indices.append(idx)
    
    if not valid_indices:
        raise ValueError("No valid start indices provided")
    
    if debug:
        print(f"\nIndex validation summary:")
        print(f"Total indices: {len(start_indices)}")
        if skipped_early > 0:
            print(f"Skipped {skipped_early} early indices (need {total_required_history} points of history)")
        if skipped_invalid > 0:
            print(f"Skipped {skipped_invalid} indices with invalid prices")
        print(f"Valid indices remaining: {len(valid_indices)}")
        print(f"Valid index range: [{min(valid_indices)}, {max(valid_indices)}]")
    
    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path,
        model_params,
        device=device
    )
    
    # Make predictions for each start index
    predictions_list = []
    for start_idx in valid_indices:
        # Get initial sequence with warmup period
        initial_sequence = price_series[max(0, start_idx - sequence_length - max_lookback):start_idx + 1]
        
        # Check if we have enough data for warmup
        if len(initial_sequence) < max_lookback + 1:
            print(f"Warning: Not enough historical data for warmup at index {start_idx}")
            continue
        
        # Get true future values if using forcing
        true_future = None
        if use_forcing:
            future_end = min(start_idx + n_steps, len(price_series))
            true_future = price_series[start_idx + 1:future_end + 1]
            if len(true_future) < n_steps:
                print(f"Warning: Not enough future data for forcing at index {start_idx}")
                continue
        
        # Make prediction
        predictions = make_autoregressive_prediction(
            model=model,
            initial_sequence=initial_sequence,
            n_steps=n_steps,
            device=device,
            return_periods=return_periods,
            sma_periods=sma_periods,
            use_multi_horizon=use_multi_horizon,
            horizon_weights=horizon_weights,
            use_forcing=use_forcing,
            forcing_halflife=forcing_halflife,
            true_future=true_future,
            debug=debug,
            stability_threshold=stability_threshold,
            dampening_factor=dampening_factor,
            use_ewma_smoothing=use_ewma_smoothing,
            ewma_alpha=ewma_alpha,
            temperature=temperature,
            use_sampling=use_sampling
        )
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
