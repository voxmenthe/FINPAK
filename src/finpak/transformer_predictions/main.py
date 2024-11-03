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
    epochs = 200
    max_checkpoints = 5
    sequence_length = 47
    batch_size = 64 #32
    patience = 11
    learning_rate = 6e-5
    checkpoint_path = None  # Set this to a checkpoint file path to resume training

    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 
        'DE', 'WMT', 'PG', 'MA', 'GM',
        'AAL', 'WBA', 'BA', 'INTC', 'LUV', 'PYPL'
    ]
    start_date = '1990-01-01'
    end_date = '2024-11-02'    

    # Download historical data for the tickers
    data_df = download_multiple_tickers(tickers, start_date, end_date)
    data_df = data_df.loc[:,'Adj Close'] # Extract from multi-index dataframe

    # Extract price series for all tickers and convert to tensors
    price_series_list = []
    for ticker in tickers:
        prices = data_df[ticker]
        price_tensor = torch.tensor(prices.to_numpy(), dtype=torch.float32)
        price_series_list.append(price_tensor)
    
    # Combine price series
    combined_prices = combine_price_series(price_series_list)
    
    # Create dataloaders with combined prices
    train_loader, val_loader, feature_names, target_names = create_dataloaders(
        prices=combined_prices,
        batch_size=batch_size,
        sequence_length=sequence_length,
        return_periods=[1, 5],
        sma_periods=[20],
        target_periods=[1, 5],
        num_workers=16
    )
    
    # Add some logging to see the effect
    print(f"Total combined series length: {len(combined_prices)}")
    print(f"Individual series lengths: {[len(p) for p in price_series_list]}")
    
    # Print dataset information
    print(f"Number of features: {len(feature_names)}")
    print(f"Features: {feature_names}")
    print(f"Number of targets: {len(target_names)}")
    print(f"Targets: {target_names}")
    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
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

    # Initialize model with correct input/output dimensions
    model = TimeSeriesDecoder(
        d_input=len(feature_names),
        n_outputs=len(target_names),
        **model_params_v1,
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
        start_epoch=start_epoch  # Pass the start epoch to resume training
    )
