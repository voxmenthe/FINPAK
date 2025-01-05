import os
import torch
import wandb
import argparse
from data_loading import create_dataloaders
from timeseries_decoder_v4 import TimeSeriesDecoder
from simple_train_v4 import train_model
import pandas as pd
from finpak.data.fetchers.yahoo import download_multiple_tickers
from finpak.transformer_predictions.preprocessing import combine_price_series
from configs import all_configs
from ticker_configs import train_tickers_v14, val_tickers_v14


def get_device():
    """Get the best available device for training."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train transformer model with specified config')
    parser.add_argument('--config', type=str, required=True, help='Configuration version (e.g., vMP004a)')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training from (optional)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--metrics-dir', type=str, default='metrics', help='Directory to save training metrics')
    args = parser.parse_args()

    # Set up device
    device = get_device()
    print(f"Using device: {device}")

    # Validate and load config
    if args.config not in all_configs:
        raise ValueError(f"Config '{args.config}' not found. Available configs: {list(all_configs.keys())}")
    
    CONFIG = all_configs[args.config]
    print(f"\nUsing config: {args.config}")

    # WANDB
    from creds import all_creds
    os.environ["WANDB_API_KEY"] = all_creds['WANDB_API_KEY']
    wandb.init(project="simple0", config=CONFIG)
    run = wandb.init(
        # Set the project where this run will be logged
        project=CONFIG['train_params']['wandb_project'],
        # Track hyperparameters and run metadata
        config={
            **CONFIG['train_params'],
            **CONFIG['model_params'],
            **CONFIG['data_params']
        },
    )

    # Set up data paths
    train_df_fname = 'TRAIN_VAL_DATA/train_df_v14.csv'
    val_df_fname = 'TRAIN_VAL_DATA/val_df_v14.csv'
    
    # Data parameters
    start_date = '1982-01-01'
    end_date = '2024-12-05'

    # Load or download data
    if not (pd.io.common.file_exists(train_df_fname) and pd.io.common.file_exists(val_df_fname)):
        print("\nDownloading data...")
        # Download and process training data
        train_df = download_multiple_tickers(train_tickers_v14, start_date, end_date)
        train_df = train_df.loc[:,'Adj Close']
        train_df.to_csv(train_df_fname)
        
        # Download and process validation data
        val_df = download_multiple_tickers(val_tickers_v14, start_date, end_date)
        val_df = val_df.loc[:,'Adj Close']
        val_df.to_csv(val_df_fname)
    else:
        print("\nLoading existing data files...")
        train_df = pd.read_csv(train_df_fname, index_col=0)
        val_df = pd.read_csv(val_df_fname, index_col=0)
        
        # Convert index to datetime
        train_df.index = pd.to_datetime(train_df.index)
        val_df.index = pd.to_datetime(val_df.index)
    
    # Forward fill missing values
    train_df = train_df.ffill()
    val_df = val_df.ffill()
    
    # Drop any remaining NaN values
    train_df = train_df.dropna(axis=0, how='any')
    val_df = val_df.dropna(axis=0, how='any')

    # Convert DataFrames to tensors
    train_price_series = [torch.tensor(train_df[ticker].values, dtype=torch.float32) 
                         for ticker in train_tickers_v14 if ticker in train_df.columns]
    val_price_series = [torch.tensor(val_df[ticker].values, dtype=torch.float32) 
                       for ticker in val_tickers_v14 if ticker in val_df.columns]

    # Combine price series
    train_prices = combine_price_series(train_price_series, debug=args.debug)
    val_prices = combine_price_series(val_price_series, debug=args.debug)

    # Create dataloaders
    train_loader, val_loader, feature_names, target_names = create_dataloaders(
        train_prices=train_prices,
        val_prices=val_prices,
        config=CONFIG,
        debug=args.debug
    )

    # Calculate input features
    num_features = len(CONFIG['data_params']['return_periods']) + len(CONFIG['data_params']['sma_periods'])
    if CONFIG['data_params'].get('use_volatility', False):
        num_features += len(CONFIG['data_params']['return_periods'])
    if CONFIG['data_params'].get('use_momentum', False):
        num_features += len(CONFIG['data_params'].get('momentum_periods', []))

    # Initialize model
    model = TimeSeriesDecoder(
        d_input=num_features,
        n_outputs=len(CONFIG['data_params']['target_periods']),
        **CONFIG['model_params']
    )

    # Load checkpoint if specified
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel size: {total_params:,} parameters")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    train_losses, val_losses, batch_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=CONFIG,
        checkpoint_dir=args.checkpoint_dir,
        metrics_dir=args.metrics_dir,
        debug=args.debug
    )

    print("\nTraining completed!")
    print(f"Final training loss: {train_losses[-1]:.7f}")
    print(f"Final validation loss: {val_losses[-1]:.7f}")
    print(f"Training metrics saved in: {args.metrics_dir}")
