import torch
from data_loading import create_dataloaders
from modern_timeseries_decoder import ModernTimeSeriesDecoder
from modern_train import train_model
import os

from finpak.data.fetchers.yahoo import download_multiple_tickers
from finpak.transformer_predictions.preprocessing import combine_price_series, normalize_features


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
    epochs = 15000
    max_checkpoints = 12
    sequence_length = 47
    batch_size = 128 #32 # try more?
    patience = 48
    learning_rate = 3e-5
    # Set this to a checkpoint file path to resume training or None to start from scratch
    checkpoint_path = None #'mpv1a_e99_valloss_0.0033764.pt' # 'mpv1a_e_77_valloss_0.0024084.pt' # 'mpv000_e245_valloss_0.0016670.pt' # None # 'checkpoints/mpv1_e_66_valloss_0.0017783.pt' # None  
    return_periods=[1, 2, 3, 4, 6, 12]
    sma_periods=[20]
    target_periods=[1, 2, 3, 4, 6, 12]
    DEBUG = True

    model_params_v0 = {
        "d_model": 512,
        "n_heads": 4,
        "n_layers": 56,
        "d_ff": 2048,
        "dropout": 0.12,
    }

    model_params_v000 = {
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 8,
        "d_ff": 128,
        "dropout": 0.05
    }

    model_params_v1 = {
        "d_model": 1024,
        "n_heads": 8,
        "n_layers": 88,
        "d_ff": 2048,
        "dropout": 0.32,
    }

    model_params_v1a = {
        "d_model": 1024,
        "n_heads": 8,
        "n_layers": 88,
        "d_ff": 3072,
        "dropout": 0.23,
    }

    model_params_v1b = {
        "d_model": 2048,
        "n_heads": 16,
        "n_layers": 88,
        "d_ff": 3072,
        "dropout": 0.28,
    }

    model_params_v1c = {
        "d_model": 2048,
        "n_heads": 32,
        "n_layers": 64,
        "d_ff": 2048,
        "dropout": 0.26,
    }

    model_params_v2 = {
        "d_model": 2048,
        "n_heads": 32,
        "n_layers": 96,
        "d_ff": 8192,
        "dropout": 0.12,
    }

    MODEL_PARAMS = model_params_v1c
    prefix = 'mpv1c'

    # Split tickers into training and validation sets
    # negative: 'WBA', 'LVS',
    # unsure: 
    train_tickers = [
        'AAPL', 'AAL', 'AMZN', 'AVGO', 'ADBE', 'AXP', 
        'BA', 'BIIB', 'CLX', 'CMG', 'CRM', 'DIS', 'DE',
        'EBAY', 'ED', 'FDX',
        'GM', 'GD', 'GDX', 'GOOGL', 'GS', 'HD',
        'IBM', 'INTC','ISRG', 
        'JNJ', 'JPM', 
        'KRE', 'KO',
        'LEN', 'LLY','LMT', 'LULU', 'LVS',
        'NOW', 'ORCL',
        'PG', 'MA', 'META', 'MGM','MS', 'MSFT', 'NVDA',
        'OXY', 'PANW',
        'LUV', 'PYPL', 
        'SBUX', 'SCHW', 'SMH',
        'TEVA', 'TGT','TOL', 'TSLA',
        'UAL', 'UNH', 'UPS',
        'WBA', 'WMT', 
    ]
    
    val_tickers = ['UAL', 'SNOW', 'CRWD', 'IBKR', 'AMD', 'COIN'] # 'FTNT', 'CRWD', 'CAVA', 'AMD', 'SNOW', 'UAL', 'DKNG',  # Validation tickers
    
    start_date = '1990-01-01'
    end_date = '2024-11-06'

    # Download and process training data
    train_df = download_multiple_tickers(train_tickers, start_date, end_date)
    train_df = train_df.loc[:,'Adj Close']
    
    # Download and process validation data
    val_df = download_multiple_tickers(val_tickers, start_date, end_date)
    val_df = val_df.loc[:,'Adj Close']

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
    model = ModernTimeSeriesDecoder(
        d_input=len(feature_names),
        d_model=MODEL_PARAMS["d_model"],
        n_heads=MODEL_PARAMS["n_heads"],
        n_layers=MODEL_PARAMS["n_layers"],
        dropout=MODEL_PARAMS["dropout"],
        n_outputs=len(target_names)
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
        prefix=prefix,
        use_mixed_precision=True if device.type == "cuda" else False  # Enable for CUDA devices only
    )
