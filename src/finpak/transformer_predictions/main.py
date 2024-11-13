import torch
from data_loading import create_dataloaders
#from timeseries_decoder import TimeSeriesDecoder
#from timeseries_decoder_v2 import TimeSeriesDecoder
from timeseries_decoder_v3 import TimeSeriesDecoder
from train import train_model
import os
import pandas as pd
from finpak.data.fetchers.yahoo import download_multiple_tickers
from finpak.transformer_predictions.preprocessing import combine_price_series, normalize_features

from configs import all_configs, train_tickers_v3, val_tickers_v3

def get_device():
  if torch.backends.mps.is_available():
      return torch.device("mps")
  elif torch.cuda.is_available():
      return torch.device("cuda")
  else:
      return torch.device("cpu")

# Use this device throughout your code
device = get_device()



if __name__ == "__main__":
    print(f"Using device: {device}")

    CONFIG = all_configs['vMS0001']
    # Set this to a checkpoint file path to resume training or None to start from scratch
    checkpoint_path = None #'checkpoints/mpv005a_v2_e123_valloss_0.0020898.pt' # 'checkpoints/mpv005_v2_e81_valloss_0.0019177.pt' # None #'mpv1a_e99_valloss_0.0033764.pt' # 'mpv1a_e_77_valloss_0.0024084.pt' # 'mpv000_e245_valloss_0.0016670.pt' # None # 'checkpoints/mpv1_e_66_valloss_0.0017783.pt' # None  

    architecture_version = '_v3'

    # Split tickers into training and validation sets
    # negative: 'WBA', 'LVS',
    # unsure: 
    train_tickers = train_tickers_v3
    val_tickers = val_tickers_v3
    train_df_fname = 'train_df_v3.csv'
    val_df_fname = 'val_df_v3.csv'
    FORCE_RELOAD = False

    epochs = CONFIG['train_params']['epochs']
    max_checkpoints = CONFIG['train_params']['max_checkpoints']
    batch_size = CONFIG['train_params']['batch_size']
    patience = CONFIG['train_params']['patience']
    learning_rate = CONFIG['train_params']['learning_rate']
    warmup_steps = CONFIG['train_params']['warmup_steps']
    decay_step_multiplier = CONFIG['train_params']['decay_step_multiplier']

    sequence_length = CONFIG['data_params']['sequence_length']
    return_periods = CONFIG['data_params']['return_periods']
    sma_periods = CONFIG['data_params']['sma_periods']
    target_periods = CONFIG['data_params']['target_periods']
    
    DEBUG = True

    MODEL_PARAMS = CONFIG['model_params']
    prefix = CONFIG['train_params']['prefix'] + f'{architecture_version}'
    
    start_date = '1990-01-01'
    end_date = '2024-11-05'

    # Check if the dataframes already exist
    if not os.path.exists(train_df_fname) or FORCE_RELOAD:
        # Download and process training data
        train_df = download_multiple_tickers(train_tickers, start_date, end_date)
        train_df = train_df.loc[:,'Adj Close']
        train_df.to_csv(train_df_fname)
    else:
        train_df = pd.read_csv(train_df_fname)
    
    if not os.path.exists(val_df_fname) or FORCE_RELOAD:
        # Download and process validation data
        val_df = download_multiple_tickers(val_tickers, start_date, end_date)
        val_df = val_df.loc[:,'Adj Close']
        val_df.to_csv(val_df_fname)
    else:
        val_df = pd.read_csv(val_df_fname)


    # Process training price series
    train_price_series = []
    for ticker in train_tickers:
        prices = train_df[ticker]
        price_tensor = torch.tensor(prices.to_numpy(), dtype=torch.float32)
        train_price_series.append(price_tensor)
    
    # Process validation price series
    val_price_series = []
    for ticker in val_tickers:
        prices = val_df[ticker]
        price_tensor = torch.tensor(prices.to_numpy(), dtype=torch.float32)
        val_price_series.append(price_tensor)
    
    # Combine price series separately for train and validation
    combined_train_prices = combine_price_series(train_price_series)
    combined_val_prices = combine_price_series(val_price_series)
    
    # Create dataloaders with separate train/val prices
    train_loader, val_loader, feature_names, target_names = create_dataloaders(
        train_prices=combined_train_prices,
        val_prices=combined_val_prices,
        batch_size=batch_size,
        sequence_length=sequence_length,
        return_periods=return_periods,
        sma_periods=sma_periods,
        target_periods=target_periods,
        num_workers=16,
        debug=DEBUG
    )
    
    if DEBUG:
        # Add logging
        print("\nDataset Information:")
        print(f"Training tickers: {train_tickers}")
        print(f"Validation tickers: {val_tickers}")
        print(f"\nTraining series length: {len(combined_train_prices)}")
        print(f"Validation series length: {len(combined_val_prices)}")
        print(f"\nNumber of features: {len(feature_names)}")
        print(f"Features: {feature_names}")
        print(f"Number of targets: {len(target_names)}")
        print(f"Targets: {target_names}")
        print(f"\nTraining batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")

    # Initialize model with correct input/output dimensions
    model = TimeSeriesDecoder(
        d_input=len(feature_names),
        n_outputs=len(target_names),
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
    
    # Train model - pass the device and start_epoch
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=epochs,
        max_checkpoints=max_checkpoints,
        patience=patience,
        device=device,
        learning_rate=learning_rate,
        start_epoch=start_epoch,
        prefix=prefix,  # Pass the start epoch to resume training,
        warmup_steps=warmup_steps,
        decay_step_multiplier=decay_step_multiplier,
    )
