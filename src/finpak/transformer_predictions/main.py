import torch
from data_loading import create_dataloaders
from timeseries_decoder import TimeSeriesDecoder
from train import train_model
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
    epochs = 260
    max_checkpoints = 8
    sequence_length = 47
    batch_size = 64 #32
    patience = 18
    learning_rate = 6e-5
    checkpoint_path = 'checkpoints/mpv1_e_42_valloss_0.0019789.pt' # None  # Set this to a checkpoint file path to resume training

    model_params_v0 = {
        "d_model": 512,
        "n_heads": 4,
        "n_layers": 56,
        "d_ff": 2048,
        "dropout": 0.12,
    }

    model_params_v1 = {
        "d_model": 1024,
        "n_heads": 8,
        "n_layers": 88,
        "d_ff": 2048,
        "dropout": 0.32,
    }

    model_params_v2 = {
        "d_model": 2048,
        "n_heads": 32,
        "n_layers": 96,
        "d_ff": 8192,
        "dropout": 0.12,
    }

    MODEL_PARAMS = model_params_v1
    prefix = 'mpv1'

    # Split tickers into training and validation sets
    train_tickers = [
        'IBM', 'JPM', 'LEN', 'GS', 'OXY', 'SCHW', 'ISRG', 'HD', 'AVGO', 'PANW',
        'ADBE', 'NOW', 'CMG', 'LVS', 'ORCL',
        'DE', 'WMT', 'PG', 'MA', 'GM', 'CLX', 'CRM', 'DIS', 'EBAY',
        'AAL', 'WBA', 'BA', 'INTC', 'LUV', 'PYPL', 'ED', 'AXP', 'GD', 'GDX',
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    ]
    
    val_tickers = ['AMD', 'UAL', 'DKNG', 'IBKR', 'SNOW'] # 'FTNT', 'CRWD',  'CAVA']  # Validation tickers
    
    start_date = '1990-01-01'
    end_date = '2024-11-02'

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
        return_periods=[1, 5],
        sma_periods=[20],
        target_periods=[1, 5],
        num_workers=16
    )
    
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
        prefix=prefix,  # Pass the start epoch to resume training
    )
