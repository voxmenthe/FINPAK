import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim
from pathlib import Path
import numpy as np
from datetime import datetime
import pandas as pd
import os
import argparse

#from timeseries_decoder_v4_mlx import TimeseriesDecoderMLX
from timeseries_decoder_v4_optimized_mlx import TimeseriesDecoderMLX
from train_v4_mlx import train, load_checkpoint
from finpak.data.fetchers.yahoo import download_multiple_tickers
from finpak.transformer_predictions.preprocessing_mlx import combine_price_series, normalize_features
from ticker_cycler import TickerCycler
from configs import all_configs
from ticker_configs import train_tickers_v11, val_tickers_v11


def mse_loss(pred: mx.array, target: mx.array) -> mx.array:
    """Mean squared error loss function for MLX."""
    return mx.mean((pred - target) ** 2)


def create_data_loaders(
    train_data: np.ndarray,
    train_targets: np.ndarray,
    val_data: np.ndarray,
    val_targets: np.ndarray,
    batch_size: int
):
    """Creates training and validation data loaders."""
    def train_loader():
        indices = np.random.permutation(len(train_data))
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            # Convert directly to MLX arrays with float32 dtype
            batch_x = mx.array(train_data[batch_indices], dtype=mx.float32)
            batch_y = mx.array(train_targets[batch_indices], dtype=mx.float32)
            yield batch_x, batch_y
    
    def val_loader():
        indices = np.arange(len(val_data))
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            # Convert directly to MLX arrays with float32 dtype
            batch_x = mx.array(val_data[batch_indices], dtype=mx.float32)
            batch_y = mx.array(val_targets[batch_indices], dtype=mx.float32)
            yield batch_x, batch_y
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train transformer model with specified config')
    parser.add_argument('--config', type=str, required=True, help='Configuration version (e.g., vMP004a)')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training from (optional)')
    args = parser.parse_args()

    if args.config not in all_configs:
        raise ValueError(f"Config '{args.config}' not found. Available configs: {list(all_configs.keys())}")
    
    CONFIG = all_configs[args.config]
    print(CONFIG)

    train_tickers = train_tickers_v11
    val_tickers = val_tickers_v11

    if CONFIG['data_params'].get('reverse_tickers', False):
        train_tickers = train_tickers[::-1]
        val_tickers = val_tickers[::-1]

    train_df_fname = 'TRAIN_VAL_DATA/train_df_v11.csv'
    val_df_fname = 'TRAIN_VAL_DATA/val_df_v11.csv'
    FORCE_RELOAD = False

    start_date = '1986-01-01'
    end_date = '2024-11-27'

    # Extract parameters from config
    batch_size = CONFIG['train_params']['batch_size']
    sequence_length = CONFIG['data_params']['sequence_length']
    return_periods = CONFIG['data_params']['return_periods']
    sma_periods = CONFIG['data_params']['sma_periods']
    target_periods = CONFIG['data_params']['target_periods']
    use_volatility = CONFIG['data_params'].get('use_volatility', False)
    use_momentum = CONFIG['data_params'].get('use_momentum', False)
    momentum_periods = CONFIG['data_params'].get('momentum_periods', [9, 28, 47])

    # Calculate total number of features
    num_features = len(return_periods) + len(sma_periods)
    if use_volatility:
        num_features += len(return_periods)
    if use_momentum:
        num_features += len(momentum_periods)

    # Load or download data
    if os.path.exists(train_df_fname) and os.path.exists(val_df_fname) and not FORCE_RELOAD:
        print("Loading existing data files...")
        train_df = pd.read_csv(train_df_fname, index_col=0)
        val_df = pd.read_csv(val_df_fname, index_col=0)
        
        # Convert index to datetime
        train_df.index = pd.to_datetime(train_df.index)
        val_df.index = pd.to_datetime(val_df.index)
        
        # Forward fill any missing values within each ticker's series
        train_df = train_df.ffill()
        val_df = val_df.ffill()
    else:
        print("Downloading data...")
        train_df = download_multiple_tickers(train_tickers, start_date, end_date)
        val_df = download_multiple_tickers(val_tickers, start_date, end_date)
        
        # Save the data
        train_df.to_csv(train_df_fname)
        val_df.to_csv(val_df_fname)

    # Process data into features
    train_features, train_targets = combine_price_series(
        train_df, sequence_length, return_periods, sma_periods, 
        target_periods, use_volatility, use_momentum, momentum_periods
    )
    val_features, val_targets = combine_price_series(
        val_df, sequence_length, return_periods, sma_periods, 
        target_periods, use_volatility, use_momentum, momentum_periods
    )

    # Normalize features
    train_features, val_features = normalize_features(train_features, val_features)

    # Convert to numpy arrays if they aren't already
    train_features = np.array(train_features)
    train_targets = np.array(train_targets)
    val_features = np.array(val_features)
    val_targets = np.array(val_targets)
    
    # Create model and move to MLX format
    model = TimeseriesDecoderMLX(
        d_input=num_features,
        d_model=CONFIG['model_params']['d_model'],
        n_heads=CONFIG['model_params']['n_heads'],
        n_layers=CONFIG['model_params']['n_layers'],
        d_ff=CONFIG['model_params']['d_ff'],
        dropout=CONFIG['model_params']['dropout'],
        n_outputs=len(target_periods),
        base=CONFIG['model_params'].get('base', 10000)
    )
    
    # Initialize optimizer
    optimizer = mlx_optim.AdamW(
        learning_rate=CONFIG['train_params']['scheduler']['base_lr'],
        weight_decay=CONFIG['train_params'].get('weight_decay', 0.0)
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_features, train_targets,
        val_features, val_targets,
        CONFIG['train_params']['batch_size']
    )
    
    # Set up ticker cyclers if enabled
    train_cycler = None
    val_cycler = None
    
    if CONFIG['train_params'].get('use_train_cycling', False):
        train_cycler = TickerCycler(
            tickers=train_tickers,
            subset_size=CONFIG['train_params']['train_subset_size'],
            overlap_size=CONFIG['train_params'].get('train_overlap_size'),
            use_anchor=CONFIG['train_params'].get('use_train_anchor', False)
        )
        print("\nInitialized training set cycler:")
        print(f"Number of training subsets: {len(train_cycler.subsets)}")
        print(f"First training subset: {train_cycler.get_current_subset()}")
    
    if CONFIG['train_params'].get('use_val_cycling', False):
        val_cycler = TickerCycler(
            tickers=val_tickers,
            subset_size=CONFIG['train_params']['val_subset_size'],
            overlap_size=CONFIG['train_params'].get('val_overlap_size'),
            use_anchor=CONFIG['train_params'].get('use_val_anchor', False)
        )
        print("\nInitialized validation set cycler:")
        print(f"Number of validation subsets: {len(val_cycler.subsets)}")
        print(f"First validation subset: {val_cycler.get_current_subset()}")
    
    # Training
    save_dir = Path(f"checkpoints/{CONFIG['train_params']['prefix']}")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=mse_loss,
        n_epochs=CONFIG['train_params']['epochs'],
        save_dir=save_dir,
        save_freq=1,
        early_stopping_patience=CONFIG['train_params']['patience'],
        min_delta=CONFIG['train_params'].get('min_delta', 1e-6),
        print_every=CONFIG['train_params'].get('print_every', 10),
        max_checkpoints=CONFIG['train_params'].get('max_checkpoints', 5),
        weight_decay=CONFIG['train_params'].get('weight_decay', 0.0),
        train_cycler=train_cycler,  # Pass cycler instead of subset
        val_cycler=val_cycler,      # Pass cycler instead of subset
        current_cycle=0,            # Start with cycle 0
        config=CONFIG
    )
    
    # Save final model
    model.save_weights(str(save_dir / "model_final.npz"))
    
    print("\nTraining completed!")
    print(f"Model saved to: {save_dir}")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")


if __name__ == "__main__":
    main()
