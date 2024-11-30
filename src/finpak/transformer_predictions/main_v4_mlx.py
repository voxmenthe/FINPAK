import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim
from pathlib import Path
import numpy as np
from datetime import datetime

from timeseries_decoder_v4_mlx import TimeseriesDecoderMLX
from train_v4_mlx import train

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
            yield train_data[batch_indices], train_targets[batch_indices]
    
    def val_loader():
        indices = np.arange(len(val_data))
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            yield val_data[batch_indices], val_targets[batch_indices]
    
    return train_loader, val_loader

def main():
    # Configuration
    config = {
        'd_input': 3,
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 3,
        'd_ff': 256,
        'dropout': 0.1,
        'n_outputs': 2,
        'use_multi_scale': True,
        'temporal_scales': [1, 2, 4],
        'base': 10000,
        'batch_size': 32,
        'n_epochs': 100,
        'learning_rate': 1e-4,
        'early_stopping_patience': 10
    }
    
    # Create model
    model = TimeseriesDecoderMLX(
        d_input=config['d_input'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        n_outputs=config['n_outputs'],
        use_multi_scale=config['use_multi_scale'],
        temporal_scales=config['temporal_scales'],
        base=config['base']
    )
    
    # Initialize optimizer
    optimizer = mlx_optim.Adam(learning_rate=config['learning_rate'])
    
    # TODO: Load your data here
    # For now using dummy data
    train_data = np.random.randn(1000, 60, config['d_input'])
    train_targets = np.random.randn(1000, config['n_outputs'])
    val_data = np.random.randn(200, 60, config['d_input'])
    val_targets = np.random.randn(200, config['n_outputs'])
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data, train_targets,
        val_data, val_targets,
        config['batch_size']
    )
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"checkpoints/run_{timestamp}")
    
    # Train model
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=mse_loss,
        n_epochs=config['n_epochs'],
        save_dir=save_dir,
        early_stopping_patience=config['early_stopping_patience']
    )
    
    # Save final model
    model.save_weights(str(save_dir / "model_final.npz"))
    
    print("Training completed!")
    print(f"Model saved to {save_dir}")

if __name__ == "__main__":
    main()
