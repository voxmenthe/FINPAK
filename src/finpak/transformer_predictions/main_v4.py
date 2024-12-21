import re, os, torch, argparse
from data_loading import create_dataloaders
from timeseries_decoder_v4 import TimeSeriesDecoder
from train import train_model
import pandas as pd
from finpak.data.fetchers.yahoo import download_multiple_tickers
from finpak.transformer_predictions.preprocessing import combine_price_series, normalize_features
from ticker_cycler import TickerCycler
from data_loading import create_dataloaders, create_subset_dataloaders
from datetime import datetime

from configs import all_configs
from ticker_configs import train_tickers_v14, val_tickers_v14


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
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training from (optional)')
    parser.add_argument('--metrics_output', type=str, default='training_metrics.csv', 
                       help='Path to save training metrics CSV (default: training_metrics.csv)')
    args = parser.parse_args()

    # Use this device throughout your code
    device = get_device()

    print(f"Using device: {device}")

    if args.config not in all_configs:
        raise ValueError(f"Config '{args.config}' not found. Available configs: {list(all_configs.keys())}")
    
    CONFIG = all_configs[args.config]
    print(CONFIG)
    
    # Use checkpoint path from command line argument
    checkpoint_path = args.checkpoint
    if checkpoint_path:
        print(f"Will attempt to load checkpoint from: {checkpoint_path}")

    checkpoint_dir = 'checkpoints'

    train_tickers = train_tickers_v14
    val_tickers = val_tickers_v14

    if CONFIG['data_params'].get('reverse_tickers', False):
        train_tickers = train_tickers[::-1]
        val_tickers = val_tickers[::-1]

    train_df_fname = 'TRAIN_VAL_DATA/train_df_v14.csv'
    val_df_fname = 'TRAIN_VAL_DATA/val_df_v14.csv'
    FORCE_RELOAD = False

    start_date = '1982-01-01'
    end_date = '2024-12-05'

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
    print(f"  - Initial train subset: {initial_train_tickers}")
    print(f"  - Initial val subset: {initial_val_tickers}")

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
    trained_subsets = []  # Track which subsets we've trained on

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Try to load metadata from new format first
            try:
                metadata = checkpoint['metadata']
                start_epoch = metadata.epoch + 1
                print(f"Resuming training from epoch {start_epoch}")
                
                # Load trained subsets history
                if hasattr(metadata, 'trained_subsets'):
                    trained_subsets = metadata.trained_subsets
                    print(f"Loaded training history for {len(trained_subsets)} subsets")
                    
                    # Set train_cycler to next subset after last trained
                    if trained_subsets:
                        while train_cycler.get_current_subset() != trained_subsets[-1]:
                            if not train_cycler.has_more_subsets():
                                print("Warning: Could not find last trained subset, starting from beginning")
                                train_cycler.reset()
                                break
                            train_cycler.next_subset()
                        # Move to next subset
                        if train_cycler.has_more_subsets():
                            train_cycler.next_subset()
                else:
                    # Try to infer from checkpoint name
                    match = re.search(r'_tc(\d+)_', checkpoint_path)
                    if match:
                        train_cycle = int(match.group(1))
                        print(f"Inferred train cycle {train_cycle} from checkpoint name")
                        # Set train_cycler to next subset
                        while train_cycler.current_subset_idx < train_cycle:
                            if not train_cycler.has_more_subsets():
                                print("Warning: Could not reach inferred train cycle, starting from beginning")
                                train_cycler.reset()
                                break
                            train_cycler.next_subset()
                        if train_cycler.has_more_subsets():
                            train_cycler.next_subset()
                    else:
                        print("No train subset history found, starting from first subset")
                
                # Print loaded metadata for debugging
                print("Loaded checkpoint metadata:")
                print(f"  - Original config: {metadata.config}")
                print(f"  - Training loss: {metadata.train_loss}")
                print(f"  - Validation loss: {metadata.val_loss}")
                print(f"  - Model parameters: {metadata.model_params}")
                print(f"  - Saved at: {metadata.timestamp}")
                print(f"  - Current train subset: {train_cycler.get_current_subset()}")
                print(f"  - Current val subset: {validation_cycler.get_current_subset()}")
            
            # Fall back to old format if metadata not found
            except (KeyError, AttributeError) as e:
                print(f"Loading checkpoint metadata (old format): {e}")
                start_epoch = checkpoint.get('epoch', 0) + 1
                print(f"Resuming training from epoch {start_epoch}")
                
                # Try to infer from checkpoint name
                match = re.search(r'_tc(\d+)_', checkpoint_path)
                if match:
                    train_cycle = int(match.group(1))
                    print(f"Inferred train cycle {train_cycle} from checkpoint name")
                    # Set train_cycler to next subset
                    while train_cycler.current_subset_idx < train_cycle:
                        if not train_cycler.has_more_subsets():
                            print("Warning: Could not reach inferred train cycle, starting from beginning")
                            train_cycler.reset()
                            break
                        train_cycler.next_subset()
                    if train_cycler.has_more_subsets():
                        train_cycler.next_subset()
        
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from epoch 0")
            start_epoch = 0
            train_cycler.reset()

    # Print model size and trainable parameters
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Initialize DataFrame to store metrics
    metrics_df = pd.DataFrame(columns=[
        'epoch', 'train_loss', 'val_loss', 
        'train_subset_tickers', 'test_subset_tickers'
    ])

    # Train model with simplified interface
    train_losses, val_losses, metrics_df = train_model(
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
        debug=DEBUG,
        trained_subsets=trained_subsets,
        metrics_df=metrics_df
    )

    # Save metrics to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_filename = f"{args.metrics_output.rsplit('.', 1)[0]}_{timestamp}.csv"
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"Training metrics saved to: {metrics_filename}")
