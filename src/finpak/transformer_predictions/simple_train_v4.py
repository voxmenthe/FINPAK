import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Any
import pandas as pd
from dataclasses import dataclass
from torch.serialization import add_safe_globals
from optimizers import CAdamW
import os
import numpy as np
from datetime import datetime

OPTIMIZER = CAdamW  # torch.optim.AdamW


@dataclass
class TrainingMetrics:
    """Simple container for training metrics"""
    epoch: int
    train_loss: float
    val_loss: float
    model_params: Dict[str, int]
    training_history: Dict[str, List[float]]
    timestamp: str

# Register our metadata class as safe for serialization
add_safe_globals({
    'TrainingMetrics': TrainingMetrics,
    'pd.Timestamp': pd.Timestamp
})

def save_metrics_to_csv(
    metrics_dir: str,
    config_name: str,
    epoch: int,
    step: int,
    train_loss: float,
    val_loss: float,
    learning_rate: float,
    timestamp: str
) -> str:
    """
    Save training metrics to a CSV file.
    
    Args:
        metrics_dir: Directory to save metrics
        config_name: Name of the configuration being used
        epoch: Current epoch number
        step: Current step number (total steps from start of training)
        train_loss: Current training loss
        val_loss: Current validation loss
        learning_rate: Current learning rate
        timestamp: Current timestamp
    
    Returns:
        Path to the metrics file
    """
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Create filename with config and timestamp
    base_timestamp = timestamp.split('T')[0]  # Get just the date part
    metrics_file = os.path.join(metrics_dir, f"training_metrics_{config_name}_{base_timestamp}.csv")
    
    # Create or load the metrics dataframe
    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)
    else:
        metrics_df = pd.DataFrame(columns=[
            'timestamp', 'epoch', 'step', 'train_loss', 'val_loss', 'learning_rate'
        ])
    
    # Add new row
    new_row = pd.DataFrame([{
        'timestamp': timestamp,
        'epoch': epoch,
        'step': step,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': learning_rate
    }])
    wandb.log({
        'step': step,
        'step_train_loss': train_loss,
        'step_val_loss': val_loss,
    })
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    metrics_df.to_csv(metrics_file, index=False)
    
    return metrics_file

def get_scheduler(optimizer, config, steps_per_epoch):
    """Create learning rate scheduler based on config."""
    scheduler_config = config['train_params']['scheduler']
    scheduler_type = scheduler_config['type']
    
    if scheduler_type == "cyclical":
        base_lr = scheduler_config['base_lr']
        max_lr = scheduler_config.get('max_lr', base_lr * 10)
        min_lr = scheduler_config.get('min_lr', base_lr / 10)
        cycle_length = scheduler_config['cycle_params']['cycle_length'] * steps_per_epoch
        cycles = scheduler_config['cycle_params'].get('cycles')
        decay_factor = scheduler_config['cycle_params'].get('decay_factor', 1.0)
        
        def lr_lambda(step):
            if cycles is not None and step >= cycle_length * cycles:
                return min_lr / base_lr
                
            current_cycle = step // cycle_length
            cycle_progress = (step % cycle_length) / cycle_length
            
            # Triangle wave function for cyclical learning rate
            if cycle_progress < 0.5:
                lr = min_lr + (max_lr - min_lr) * (2 * cycle_progress)
            else:
                lr = max_lr - (max_lr - min_lr) * (2 * (cycle_progress - 0.5))
            
            # Apply decay to peak learning rate if specified
            if decay_factor != 1.0:
                lr = min_lr + (lr - min_lr) * (decay_factor ** current_cycle)
            
            return lr / base_lr
            
    elif scheduler_type == "warmup_decay":
        base_lr = scheduler_config['base_lr']
        max_lr = scheduler_config.get('max_lr', base_lr)
        min_lr = scheduler_config.get('min_lr', base_lr / 100)
        warmup_steps = scheduler_config['warmup_epochs'] * steps_per_epoch
        total_steps = config['train_params']['epochs'] * steps_per_epoch
        
        def lr_lambda(step):
            if step < warmup_steps:
                return min_lr + (max_lr - min_lr) * (step / warmup_steps) / base_lr
            else:
                # Cosine decay from max_lr to min_lr
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + np.cos(progress * np.pi))
                return (min_lr + (max_lr - min_lr) * cosine_decay) / base_lr
                
    elif scheduler_type == "one_cycle":
        base_lr = scheduler_config['base_lr']
        max_lr = scheduler_config.get('max_lr', base_lr * 10)
        min_lr = scheduler_config.get('min_lr', base_lr / 10)
        warmup_steps = scheduler_config['warmup_epochs'] * steps_per_epoch
        total_steps = config['train_params']['epochs'] * steps_per_epoch
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return (min_lr + (max_lr - min_lr) * (step / warmup_steps)) / base_lr
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return (min_lr + (max_lr - min_lr) * (0.5 * (1 + np.cos(progress * np.pi)))) / base_lr
    

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    config: Dict[str, Any] = None,
    checkpoint_dir: str = "checkpoints",
    metrics_dir: str = "metrics",
    debug: bool = False
) -> Tuple[List[float], List[float], List[float]]:
    """
    Simple training function that tracks training and validation loss.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on
        config: Configuration dictionary
        checkpoint_dir: Directory to save checkpoints
        metrics_dir: Directory to save metrics CSV files
        debug: Whether to print debug information
    
    Returns:
        Tuple of (train_losses, val_losses, batch_losses) lists
    """
    if config is None:
        raise ValueError("Config must be provided")

    # Extract parameters from config
    train_params = config['train_params']
    n_epochs = train_params['epochs']
    weight_decay = train_params.get('weight_decay', 0.1)
    print_every = train_params.get('print_every', 1)
    save_metrics_every = train_params.get('save_metrics_every', 100)  # Save metrics every N steps

    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = OPTIMIZER(
        model.parameters(),
        lr=config['train_params']['scheduler']['base_lr'],
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )
    
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    
    train_losses = []
    val_losses = []
    batch_losses = []  # Track individual batch losses
    total_steps = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Debug prints for input data
            if debug:
                print(f"\nBatch {batch_idx} stats:")
                print(f"Input shape: {batch_x.shape}")
                print(f"Target shape: {batch_y.shape}")
                print(f"Input stats: min={batch_x.min().item():.4f}, max={batch_x.max().item():.4f}")
                print(f"Target stats: min={batch_y.min().item():.4f}, max={batch_y.max().item():.4f}")
            
            predictions = model(batch_x)
            loss = F.mse_loss(predictions, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)
            
            # Save metrics periodically
            if total_steps % save_metrics_every == 0:
                # Calculate validation loss
                model.eval()
                current_val_loss = 0.0
                with torch.no_grad():
                    for val_batch_x, val_batch_y in val_loader:
                        val_batch_x = val_batch_x.to(device)
                        val_batch_y = val_batch_y.to(device)
                        val_predictions = model(val_batch_x)
                        current_val_loss += F.mse_loss(val_predictions, val_batch_y).item()
                current_val_loss /= len(val_loader)
                
                # Save metrics
                metrics_file = save_metrics_to_csv(
                    metrics_dir=metrics_dir,
                    config_name=config['train_params']['prefix'],
                    epoch=epoch,
                    step=total_steps,
                    train_loss=batch_loss,
                    val_loss=current_val_loss,
                    learning_rate=scheduler.get_last_lr()[0],
                    timestamp=datetime.now().isoformat()
                )
                
                if debug:
                    print(f"\nSaved metrics to {metrics_file}")
                
                # Switch back to training mode
                model.train()
            
            # Print batch progress
            if debug and (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} - Batch {batch_idx+1}/{len(train_loader)}")
                print(f"Batch loss: {batch_loss:.7f}")
                print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
            
            wandb.log({
                'epoch': epoch,
                'total_steps': total_steps,
                'train_loss': batch_loss,
                'val_loss': current_val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            total_steps += 1
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                predictions = model(batch_x)
                val_loss += F.mse_loss(predictions, batch_y).item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Save checkpoint
        if (epoch + 1) % train_params.get('save_every', 5) == 0:
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_losses[-1],
                val_loss=val_losses[-1],
                model_params={
                    'total_params': sum(p.numel() for p in model.parameters()),
                    'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                },
                training_history={
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'batch_losses': batch_losses
                },
                timestamp=datetime.now().isoformat()
            )
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }
            
            checkpoint_name = f"{config['train_params']['prefix']}_e{epoch+1}_valloss_{val_loss:.7f}.pt"
            torch.save(checkpoint, os.path.join(checkpoint_dir, checkpoint_name))
            
            if debug:
                print(f"\nSaved checkpoint: {checkpoint_name}")
        
        # Print epoch progress
        if (epoch + 1) % print_every == 0:
            print(f"\nEpoch: {epoch+1}/{n_epochs}")
            print(f"Training loss: {train_losses[-1]:.7f}")
            print(f"Validation loss: {val_losses[-1]:.7f}")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
    
    # Save final checkpoint
    metrics = TrainingMetrics(
        epoch=n_epochs-1,
        train_loss=train_losses[-1],
        val_loss=val_losses[-1],
        model_params={
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        training_history={
            'train_losses': train_losses,
            'val_losses': val_losses,
            'batch_losses': batch_losses
        },
        timestamp=datetime.now().isoformat()
    )
    
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    final_checkpoint_name = f"{config['train_params']['prefix']}_final_e{n_epochs}_valloss_{val_losses[-1]:.7f}.pt"
    torch.save(final_checkpoint, os.path.join(checkpoint_dir, final_checkpoint_name))
    print(f"\nSaved final checkpoint: {final_checkpoint_name}")
    
    # Save final metrics
    final_metrics_file = save_metrics_to_csv(
        metrics_dir=metrics_dir,
        config_name=config['train_params']['prefix'],
        epoch=n_epochs-1,
        step=total_steps,
        train_loss=train_losses[-1],
        val_loss=val_losses[-1],
        learning_rate=scheduler.get_last_lr()[0],
        timestamp=datetime.now().isoformat()
    )
    print(f"Final metrics saved to: {final_metrics_file}")
    
    return train_losses, val_losses, batch_losses
