import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from torch.utils.data import DataLoader
import os
from heapq import heappush, heappushpop
from early_stopping import EarlyStopping



def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    warmup_steps: int = 1000,
    decay_step_multiplier: Optional[int] = None,
    max_checkpoints: int = 3,
    checkpoint_dir: str = "checkpoints",
    prefix: str = 'mpv',
    patience: int = 7,
    min_delta: float = 0.0,
    start_epoch: int = 0,
    config: Optional[dict] = None
) -> Tuple[List[float], List[float]]:
    """
    Train the model with learning rate warmup and save the best checkpoints
    
    Args:
        start_epoch (int): Epoch to start/resume training from (default: 0)
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    if config is not None:
        weight_decay = config['train_params']['weight_decay']
    else:
        weight_decay = 0.1

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
                # Linear warmup to 1.0 at peak_step
                return step / peak_step
            else:
                # Cosine decay over fixed period after peak
                steps_after_peak = step - peak_step
                progress = min(1.0, steps_after_peak / decay_steps)  # clamp to 1.0 max
                return 0.5 * (1 + np.cos(progress * np.pi))
    else:
        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
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
                    if f'{prefix}_e{worst_epoch}_' in filename:
                        os.remove(os.path.join(checkpoint_dir, filename))
                
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
        
        if (epoch + 1) % 10 == 0:
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
