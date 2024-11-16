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
import random


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
    """
    if config is None:
        raise ValueError("Config must be provided")

    # Set random seed if provided
    if 'seed' in config['train_params']:
        seed_everything(config['train_params']['seed'])

    # Extract all parameters from config
    train_params = config['train_params']
    n_epochs = train_params['epochs']
    learning_rate = train_params['learning_rate']
    initial_learning_rate = train_params.get('initial_learning_rate', learning_rate / 100)
    warmup_steps = train_params['warmup_steps']
    decay_step_multiplier = train_params.get('decay_step_multiplier')
    max_checkpoints = train_params['max_checkpoints']
    prefix = train_params['prefix']
    patience = train_params['patience']
    weight_decay = train_params.get('weight_decay', 0.1)
    
    # Learning rate adaptation parameters
    enable_lr_adaptation = train_params.get('enable_lr_adaptation', False)
    lr_acceleration_factor = train_params.get('lr_acceleration_factor', 1.2)
    lr_deceleration_factor = train_params.get('lr_deceleration_factor', 0.8)
    lr_adaptation_epochs = train_params.get('lr_adaptation_epochs', 5)
    min_lr = train_params.get('min_lr')
    max_lr = train_params.get('max_lr')
    min_delta = train_params.get('min_delta', 0.0)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Calculate the minimum steps before early stopping can trigger
    steps_per_epoch = len(train_loader)
    min_steps_before_stopping = warmup_steps

    if decay_step_multiplier:
        # Add half of the decay period
        min_steps_before_stopping += (warmup_steps * decay_step_multiplier) // 2
    
    min_epochs_before_stopping = min_steps_before_stopping // steps_per_epoch

    # Initialize early stopping with the minimum epochs requirement
    early_stop = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        max_checkpoints=max_checkpoints,
        min_epochs=min_epochs_before_stopping
    )
    
    # Initialize heap for keeping track of best models (max heap using negative loss)
    best_models = []

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )

    if decay_step_multiplier:
        # Learning rate scheduler with fixed decay period
        def lr_lambda(step):
            peak_step: int = warmup_steps
            # Define decay period as multiple of warmup period (e.g., 10x longer than warmup)
            decay_steps: int = int(peak_step * decay_step_multiplier)  
            
            if step < peak_step:
                # Linear warmup from initial_learning_rate to learning_rate
                warmup_factor = step / peak_step
                return (initial_learning_rate + (learning_rate - initial_learning_rate) * warmup_factor) / learning_rate
            else:
                # Cosine decay over fixed period after peak
                steps_after_peak = step - peak_step
                progress = min(1.0, steps_after_peak / decay_steps)
                return 0.5 * (1 + np.cos(progress * np.pi))
    else:
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup from initial_learning_rate to learning_rate
                warmup_factor = step / warmup_steps
                return (initial_learning_rate + (learning_rate - initial_learning_rate) * warmup_factor) / learning_rate
            return 0.5 * (1 + np.cos((step - warmup_steps) * np.pi / n_epochs))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
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
        
        # Adaptive learning rate adjustment
        if enable_lr_adaptation and epoch >= min_epochs_before_stopping:
            improvement_status = early_stop.get_improvement_status()
            epochs_since_improvement = improvement_status['epochs_since_improvement']
            
            if epochs_since_improvement >= lr_adaptation_epochs:
                # Get current learning rate
                current_lr = scheduler.get_last_lr()[0]
                
                # Calculate new learning rate
                if epochs_since_improvement == lr_adaptation_epochs:
                    # First time we hit the adaptation threshold - decelerate
                    new_lr = current_lr * lr_deceleration_factor
                else:
                    # Further deceleration if still no improvement
                    new_lr = current_lr * lr_deceleration_factor
            else:
                # We've seen recent improvement - accelerate
                current_lr = scheduler.get_last_lr()[0]
                new_lr = current_lr * lr_acceleration_factor
            
            # Apply min/max bounds if specified
            if min_lr is not None:
                new_lr = max(min_lr, new_lr)
            if max_lr is not None:
                new_lr = min(max_lr, new_lr)
            
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        
        # Save checkpoint if it's among the best
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': current_val_loss,
            'train_loss': train_losses[-1]
        }
        
        # Handle model checkpointing
        if len(best_models) < max_checkpoints:
            heappush(best_models, (-current_val_loss, epoch))
            checkpoint_name = f'{prefix}_e{epoch}_valloss_{current_val_loss:.7f}.pt'
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
                    if f'{prefix}_e{worst_epoch}_' in filename:
                        try:
                            os.remove(os.path.join(checkpoint_dir, filename))
                        except FileNotFoundError:
                            print(f"Warning: Could not delete checkpoint file {filename}")
                
                # Save the new checkpoint
                checkpoint_name = f'{prefix}_e{epoch}_valloss_{current_val_loss:.7f}.pt'
                torch.save(checkpoint, os.path.join(checkpoint_dir, checkpoint_name))
                print(f"Saved new checkpoint with filename {checkpoint_name}")
        
        # Check early stopping
        if early_stop(current_val_loss):
            print(f"\nEarly stopping triggered after epoch {epoch}")
            # Save final checkpoint
            final_checkpoint_name = f'{prefix}_final_e{epoch}_valloss_{current_val_loss:.7f}.pt'
            torch.save(checkpoint, os.path.join(checkpoint_dir, final_checkpoint_name))
            print(f"Saved final checkpoint with filename {final_checkpoint_name}")
            break
        
        if (epoch + 1) % config['train_params']['print_every'] == 0:
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
