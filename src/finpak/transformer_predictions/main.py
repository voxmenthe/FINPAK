import torch
from data_loading import create_dataloaders
#from timeseries_decoder import TimeSeriesDecoder
#from timeseries_decoder_v2 import TimeSeriesDecoder
from timeseries_decoder_v3 import TimeSeriesDecoder
from train import train_model
import os
import pandas as pd
import argparse
from finpak.data.fetchers.yahoo import download_multiple_tickers
from finpak.transformer_predictions.preprocessing import combine_price_series, normalize_features
from ticker_cycler import TickerCycler
from data_loading import create_dataloaders, create_subset_dataloaders

from configs import all_configs
from ticker_configs import train_tickers_v8, val_tickers_v8


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train transformer model with specified config')
    parser.add_argument('--config', type=str, required=True, help='Configuration version (e.g., vMP004a)')
    args = parser.parse_args()

    # Use this device throughout your code
    device = get_device()

    print(f"Using device: {device}")

    if args.config not in all_configs:
        raise ValueError(f"Config '{args.config}' not found. Available configs: {list(all_configs.keys())}")
    
    CONFIG = all_configs[args.config]
    print(CONFIG)
    # Set this to a checkpoint file path to resume training or None to start from scratch
    checkpoint_path = None  # 'checkpoints/mpv005a_v2_e123_valloss_0.0020898.pt' # 'checkpoints/mpv005_v2_e81_valloss_0.0019177.pt' # None #'mpv1a_e99_valloss_0.0033764.pt' # 'mpv1a_e_77_valloss_0.0024084.pt' # 'mpv000_e245_valloss_0.0016670.pt' # None # 'checkpoints/mpv1_e_66_valloss_0.0017783.pt' # None  

    checkpoint_dir = 'checkpoints'

    train_tickers = train_tickers_v8
    val_tickers = val_tickers_v8

    if CONFIG['data_params'].get('reverse_tickers', False):
        train_tickers = train_tickers[::-1]
        val_tickers = val_tickers[::-1]

    train_df_fname = 'train_df_v8.csv'
    val_df_fname = 'val_df_v8.csv'
    FORCE_RELOAD = False

    # Extract only parameters needed for data loading and model initialization
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

    # Validation cycling parameters
    validation_subset_size = CONFIG['train_params']['validation_subset_size']  # Number of tickers in each validation subset
    validation_overlap = CONFIG['train_params']['validation_overlap']  # Number of tickers to overlap between subsets

    DEBUG = False
    MODEL_PARAMS = CONFIG['model_params']

    # Training cycling parameters
    train_subset_size = CONFIG['train_params']['train_subset_size']
    train_overlap = CONFIG['train_params']['train_overlap']

    start_date = '1990-01-01'
    end_date = '2024-11-15'

    # Load or download data
    if os.path.exists(train_df_fname) and os.path.exists(val_df_fname) and not FORCE_RELOAD:
        print("Loading existing data files...")
        train_df = pd.read_csv(train_df_fname, index_col=0)
        val_df = pd.read_csv(val_df_fname, index_col=0)
        
        # Convert index to datetime
        train_df.index = pd.to_datetime(train_df.index)
        val_df.index = pd.to_datetime(val_df.index)
        
        # Forward fill any missing values within each ticker's series
        train_df = train_df.ffill() # fillna(method='ffill')
        val_df = val_df.ffill() # fillna(method='ffill')
        
        # Drop any remaining NaN values
        train_df = train_df.dropna(axis=0, how='any')
        val_df = val_df.dropna(axis=0, how='any')
        
        if DEBUG:
            print("\nData loading statistics:")
            print("Training data:")
            print(f"Shape: {train_df.shape}")
            print(f"Date range: {train_df.index[0]} to {train_df.index[-1]}")
            print(f"NaN counts:\n{train_df.isna().sum()}")
            print("\nValidation data:")
            print(f"Shape: {val_df.shape}")
            print(f"Date range: {val_df.index[0]} to {val_df.index[-1]}")
            print(f"NaN counts:\n{val_df.isna().sum()}")
        
        # Debug: Print available tickers
        print("\nAvailable tickers in training data:", sorted(train_df.columns.tolist()))
        missing_train = sorted(set(train_tickers) - set(train_df.columns))
        print("\nMissing tickers in training data:", missing_train)
        print("\nAvailable tickers in validation data:", sorted(val_df.columns.tolist()))
        missing_val = sorted(set(val_tickers) - set(val_df.columns))
        print("\nMissing tickers in validation data:", missing_val)

        # Remove missing tickers from lists
        if missing_train:
            print(f"\nRemoving {len(missing_train)} missing tickers from training set")
            train_tickers = [t for t in train_tickers if t not in missing_train]
        
        if missing_val:
            print(f"\nRemoving {len(missing_val)} missing tickers from validation set")
            val_tickers = [t for t in val_tickers if t not in missing_val]

    else:
        # Download and process training data
        train_df = download_multiple_tickers(train_tickers, start_date, end_date)
        train_df = train_df.loc[:,'Adj Close']
        train_df.to_csv(train_df_fname)
        
        # Download and process validation data
        val_df = download_multiple_tickers(val_tickers, start_date, end_date)
        val_df = val_df.loc[:,'Adj Close']
        val_df.to_csv(val_df_fname)

    # Create cyclers with filtered ticker lists
    validation_cycler = TickerCycler(
        tickers=val_tickers,
        subset_size=validation_subset_size,
        overlap_size=validation_overlap,
        reverse_tickers=CONFIG['data_params'].get('reverse_tickers', False),
        use_anchor=CONFIG['data_params'].get('use_anchor', False)
    )

    train_cycler = TickerCycler(
        tickers=train_tickers,
        subset_size=train_subset_size,
        overlap_size=train_overlap,
        reverse_tickers=CONFIG['data_params'].get('reverse_tickers', False),
        use_anchor=CONFIG['data_params'].get('use_anchor', False)
    )

    # Get initial subsets from cyclers
    initial_train_tickers = train_cycler.get_current_subset()
    initial_val_tickers = validation_cycler.get_current_subset()

    # Create initial dataloaders
    train_loader, val_loader = create_subset_dataloaders(
        train_df=train_df,
        val_df=val_df,
        train_tickers=initial_train_tickers,
        val_tickers=initial_val_tickers,
        config=CONFIG,
        debug=DEBUG
    )

    # Initialize model with correct input/output dimensions
    model = TimeSeriesDecoder(
        d_input=num_features,
        n_outputs=len(target_periods),
        **MODEL_PARAMS,
    )

    # Load checkpoint if specified
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    # Print model size and trainable parameters
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train model with simplified interface
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        start_epoch=start_epoch,
        config=CONFIG,
        checkpoint_dir=checkpoint_dir,
        validation_cycler=validation_cycler,
        train_cycler=train_cycler,
        train_df=train_df,
        val_df=val_df,
        debug=DEBUG
    )
