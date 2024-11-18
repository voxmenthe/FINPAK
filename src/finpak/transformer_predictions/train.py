# flake8: noqa E501
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from torch.utils.data import DataLoader
import os
from heapq import heappush, heappushpop
from early_stopping import EarlyStopping
from ticker_cycler import TickerCycler
import random
from data_loading import create_subset_dataloaders
import pandas as pd


def seed_everything(seed: int) -> None:
    """
    Set random seed for reproducibility across all relevant libraries and systems.
    
    Args:
        seed: Integer seed for random number generation
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Additional platform-specific settings for determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    start_epoch: int = 0,
    config: Optional[dict] = None,
    checkpoint_dir: str = "checkpoints",
    validation_cycler: Optional[TickerCycler] = None,
    train_cycler: Optional[TickerCycler] = None,
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    batch_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    return_periods: Optional[List[int]] = None,
    sma_periods: Optional[List[int]] = None,
    target_periods: Optional[List[int]] = None,
    debug: bool = False
) -> Tuple[List[float], List[float]]:
    """
    Train the model using parameters from config
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on
        start_epoch: Epoch to start/resume training from
        config: Configuration dictionary containing all training parameters
        checkpoint_dir: Directory to save checkpoints
        validation_cycler: Optional TickerCycler for validation set cycling
        train_cycler: Optional TickerCycler for training set cycling
        train_df: DataFrame containing training data
        val_df: DataFrame containing validation data
        batch_size: Batch size for dataloaders
        sequence_length: Length of sequences for transformer
        return_periods: List of periods for calculating returns
        sma_periods: List of periods for calculating SMAs
        target_periods: List of periods for target returns
        debug: Whether to print debug information
    """
    if config is None:
        raise ValueError("Config must be provided")

    # Set random seed if provided
    if 'seed' in config['train_params']:
        seed_everything(config['train_params']['seed'])

    # Extract all parameters from config
    train_params = config['train_params']
    n_epochs = train_params['epochs']
    weight_decay = train_params.get('weight_decay', 0.1)
    
    # Validation cycling parameters
    enable_validation_cycling = validation_cycler is not None
    
    # Training cycling parameters
    enable_train_cycling = train_cycler is not None
    
    min_delta = train_params.get('min_delta', 0.0)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Calculate the minimum steps before early stopping can trigger
    steps_per_epoch = len(train_loader)
    min_steps_before_stopping = train_params.get('min_steps_before_stopping', 0)

    if config['train_params'].get('min_epochs_before_stopping', None) is not None:
        min_epochs_before_stopping = config['train_params']['min_epochs_before_stopping']
    else:
        min_epochs_before_stopping = min_steps_before_stopping // steps_per_epoch

    # Initialize early stopping with the minimum epochs requirement
    early_stop = EarlyStopping(
        patience=train_params['patience'],
        min_delta=min_delta,
        max_checkpoints=train_params['max_checkpoints'],
        min_epochs=min_epochs_before_stopping
    )
    
    # Initialize heap for keeping track of best models (max heap using negative loss)
    best_models = []

    model = model.to(device)
    
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

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train_params']['scheduler']['base_lr'],
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )
    
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    
    train_losses = []
    val_losses = []
    
    # Adjust total epochs to account for start_epoch
    remaining_epochs = n_epochs - start_epoch
    
    for epoch in range(start_epoch, n_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            predictions = model(batch_x)
            loss = F.mse_loss(predictions, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
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
                
        current_val_loss = val_loss / len(val_loader)
        val_losses.append(current_val_loss)
        
        # Save checkpoint if it's among the best
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': current_val_loss,
            'train_loss': train_losses[-1]
        }
        
        # Handle model checkpointing
        if len(best_models) < train_params['max_checkpoints']:
            heappush(best_models, (-current_val_loss, epoch))
            checkpoint_name = f"{config['train_params']['prefix']}_id_{config['train_params']['run_id']}_arc_{config['train_params']['architecture_version']}_e{epoch}_valloss_{current_val_loss:.7f}.pt"
            torch.save(checkpoint, os.path.join(checkpoint_dir, checkpoint_name))
        else:
            # If current model is better than worst of our best models
            if -current_val_loss > best_models[0][0]:
                # Remove the worst checkpoint file
                worst_loss, worst_epoch = heappushpop(best_models, (-current_val_loss, epoch))
                # Find and remove the old checkpoint file that matches the epoch
                for filename in os.listdir(checkpoint_dir):
                    # Skip macOS hidden files and system files
                    if filename.startswith('.'):
                        continue
                    if f"{config['train_params']['prefix']}_id_{config['train_params']['run_id']}_arc_{config['train_params']['architecture_version']}_e{worst_epoch}_" in filename:
                        try:
                            os.remove(os.path.join(checkpoint_dir, filename))
                        except FileNotFoundError:
                            print(f"Warning: Could not delete checkpoint file {filename}")
                
                # Save the new checkpoint
                checkpoint_name = f"{config['train_params']['prefix']}_id_{config['train_params']['run_id']}_arc_{config['train_params']['architecture_version']}_e{epoch}_valloss_{current_val_loss:.7f}.pt"
                torch.save(checkpoint, os.path.join(checkpoint_dir, checkpoint_name))
                print(f"Saved new checkpoint with filename {checkpoint_name}")
        
        # Check early stopping
        if early_stop(current_val_loss):
            print(f"Early stopping triggered after epoch {epoch}")
            should_continue = False
            
            if enable_train_cycling and train_cycler.has_more_subsets():
                print(f"Switching to next training subset")
                train_cycler.next_subset()
                
                # Reset early stopping when cycling to new training subset
                early_stop = EarlyStopping(
                    patience=train_params['patience'],
                    min_delta=min_delta,
                    max_checkpoints=train_params['max_checkpoints'],
                    min_epochs=min_epochs_before_stopping
                )
                
                # Move to next training subset
                train_loader = create_subset_dataloaders(
                    train_df=train_df,
                    val_df=val_df,
                    train_tickers=train_cycler.get_current_subset(),
                    val_tickers=validation_cycler.get_current_subset(),
                    config=config,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    return_periods=return_periods,
                    sma_periods=sma_periods,
                    target_periods=target_periods,
                    debug=debug
                )[0]  # Only need train_loader
                
                should_continue = True
                
                # If validation cycler is exhausted but we have more training subsets,
                # reset validation cycler to start from beginning
                if enable_validation_cycling and not validation_cycler.has_more_subsets():
                    print(f"\nResetting validation cycler to beginning as training continues")
                    validation_cycler.reset()
                    
                    # Reset early stopping for new validation cycle
                    early_stop = EarlyStopping(
                        patience=train_params['patience'],
                        min_delta=min_delta,
                        max_checkpoints=train_params['max_checkpoints'],
                        min_epochs=min_epochs_before_stopping
                    )
                    
                    val_loader = create_subset_dataloaders(
                        train_df=train_df,
                        val_df=val_df,
                        train_tickers=train_cycler.get_current_subset(),
                        val_tickers=validation_cycler.get_current_subset(),
                        config=config,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        return_periods=return_periods,
                        sma_periods=sma_periods,
                        target_periods=target_periods,
                        debug=debug
                    )[1]  # Only need val_loader
            
            elif enable_validation_cycling and validation_cycler.has_more_subsets():
                # Move to next validation subset
                print(f"\nSwitching to next validation subset after epoch {epoch}")
                validation_cycler.next_subset()
                
                # Reset early stopping when cycling to new validation subset
                early_stop = EarlyStopping(
                    patience=train_params['patience'],
                    min_delta=min_delta,
                    max_checkpoints=train_params['max_checkpoints'],
                    min_epochs=min_epochs_before_stopping
                )
                
                val_loader = create_subset_dataloaders(
                    train_df=train_df,
                    val_df=val_df,
                    train_tickers=train_cycler.get_current_subset(),
                    val_tickers=validation_cycler.get_current_subset(),
                    config=config,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    return_periods=return_periods,
                    sma_periods=sma_periods,
                    target_periods=target_periods,
                    debug=debug
                )[1]  # Only need val_loader

                should_continue = True

            if should_continue:
                # Reset best loss tracking for the new subset
                early_stop.best_loss = None
                early_stop.counter = 0
                early_stop.early_stop = False

            else:
                print(f"\nEarly stopping triggered after epoch {epoch}")
                # Save final checkpoint
                final_checkpoint_name = f"{config['train_params']['prefix']}_final_id_{config['train_params']['run_id']}_arc_{config['train_params']['architecture_version']}_e{epoch}_valloss_{current_val_loss:.7f}.pt"
                torch.save(checkpoint, os.path.join(checkpoint_dir, final_checkpoint_name))
                print(f"Saved final checkpoint with filename {final_checkpoint_name}")
                break
        
        if (epoch + 1) % train_params['print_every'] == 0:
            current_step = epoch * len(train_loader)  # Calculate current step
            print(f"Epoch: {epoch+1}/{n_epochs} (Step: {current_step})")
            print(f"Training loss: {train_losses[-1]:.7f}")
            print(f"Validation loss: {val_losses[-1]:.7f}")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
            
    # At the end of training, print information about best checkpoints
    print("\nBest checkpoints:")
    for neg_loss, epoch in sorted(best_models, reverse=True):
        print(f"Epoch {epoch}: validation loss = {-neg_loss:.7f}")
            
    return train_losses, val_losses
