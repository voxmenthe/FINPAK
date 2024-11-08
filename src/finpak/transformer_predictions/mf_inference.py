import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
from mf_transformer import MF_TimeseriesDecoder
from mf_preprocessing import create_stock_features
from preprocessing import normalize_features


def de_discretize_predictions(
    model_outputs: torch.Tensor,
    bin_edges: torch.Tensor
) -> torch.Tensor:
    """
    Convert the model's output probabilities over bins back to continuous values.

    Args:
    model_outputs: Logits over bins from the model (before softmax).
    bin_edges: Edges of the bins used during discretization.

    Returns:
    Continuous predictions.
    """
    # Apply softmax to get probabilities
    probs = F.softmax(model_outputs, dim=-1)  # Shape: (batch_size, num_bins)

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Shape: (num_bins,)

    # Compute expected value
    predictions = torch.matmul(probs, bin_centers)

    return predictions  # Shape: (batch_size,)



def make_autoregressive_prediction(
    model: MF_TimeseriesDecoder,
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
    config: dict = None,
    debug: bool = False,
) -> torch.Tensor:
    """
    Make autoregressive predictions starting from an initial sequence.
    
    Args:
        use_forcing: Whether to use exponentially decaying teacher forcing
        forcing_halflife: Number of steps over which forcing influence reduces by half
        true_future: Optional tensor of true future values for forcing (must be at least n_steps long)
    """

    bin_edges = config['fourier_params']['bin_edges']
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
                bin_edges=bin_edges,
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
            logits = model(input_sequence)
            continuous_pred = de_discretize_predictions(logits, bin_edges)
            next_return = continuous_pred.item()
          
            # Calculate next price
            last_price = current_sequence[-1].item()
            next_price = last_price * (1 + next_return)
          
            # Update price sequence
            current_sequence = torch.cat([current_sequence, torch.tensor([next_price], device=device)])
            predictions.append(next_return)

            # Update price sequence based on prediction
            last_price = current_sequence[-1].item()
            
            if use_multi_horizon:
                # Simple approach: Just use the 1-day prediction and scale down the N-day prediction
                one_day_return = logits[0][0].item()
                n_day_return = logits[0][1].item()  # Assuming second prediction is N-day return
                scaled_n_day = n_day_return / return_periods[1]  # Simple linear scaling
                
                if debug:  # Print for all steps when debugging
                    print(f"\nStep {step} diagnostics:")
                    print(f"Raw model predictions: {logits}")
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
                next_return = logits[0][0].item()
            
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
