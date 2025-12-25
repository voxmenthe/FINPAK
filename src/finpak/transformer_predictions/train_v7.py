# flake8: noqa E501
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tempfile
from pathlib import Path
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from data_loading_v7 import create_subset_dataloaders
from early_stopping import EarlyStopping
from ticker_cycler import TickerCycler
import random
import signal
import sys
from dataclasses import dataclass
from torch.serialization import add_safe_globals
from optimizers import CAdamW
import time
import shutil
import glob
import heapq

OPTIMIZER = CAdamW # torch.optim.AdamW

@dataclass
class TrainingMetadata:
    epoch: int
    val_loss: float
    train_loss: float
    train_cycle: int
    val_cycle: int
    config: Dict[str, Any]
    model_params: Dict[str, int]
    training_history: Dict[str, List[float]]
    timestamp: str
    trained_subsets: List[List[str]]

# Register our metadata class as safe for serialization
add_safe_globals({
    'TrainingMetadata': TrainingMetadata,
    'pd.Timestamp': pd.Timestamp
})

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
    debug: bool = False,
    trained_subsets: Optional[List[List[str]]] = None
) -> Tuple[List[float], List[float]]:
    """
    Train the model using parameters from config
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on
        start_epoch: Epoch to start/resume training from
        config: Configuration dictionary containing loss weights
        checkpoint_dir: Directory to save checkpoints
        validation_cycler: Optional TickerCycler for validation set cycling
        train_cycler: Optional TickerCycler for training set cycling
        train_df: DataFrame containing training data
        val_df: DataFrame containing validation data
        debug: Whether to print debug information
        trained_subsets: Optional list of previously trained subsets
    """
    if config is None:
        raise ValueError("Config must be provided")

    if trained_subsets is None:
        trained_subsets = []

    # Set random seed if provided
    if 'seed' in config['train_params']:
        seed_everything(config['train_params']['seed'])

    # Extract training parameters from config
    train_params = config['train_params']
    num_epochs = train_params['epochs']
    
    # Get learning rate parameters from scheduler config
    scheduler_config = train_params.get('scheduler', {})
    learning_rate = scheduler_config.get('base_lr', 1e-4)  # Default to 1e-4 if not specified
    weight_decay = scheduler_config.get('weight_decay', 0.01)
    warmup_epochs = scheduler_config.get('warmup_epochs', 6)
    warmup_steps = warmup_epochs * len(train_loader)  # Convert epochs to steps
    patience = train_params.get('patience', 5)
    min_epochs = train_params.get('min_epochs', 10)
    max_epochs = train_params.get('max_epochs', float('inf'))
    gradient_clip = train_params.get('gradient_clip', 1.0)
    
    # Create optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = OPTIMIZER(optimizer_grouped_parameters, lr=learning_rate)
    
    # Initialize scheduler
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    
    # Initialize early stopping with the minimum epochs requirement
    early_stopping = EarlyStopping(
        patience=patience,
        min_epochs=min_epochs,
        min_delta=train_params.get('min_delta', 0)
    )

    # Initialize training history
    train_losses = []
    val_losses = []
    train_loss_components = {'continuous': [], 'categorical': []}
    val_loss_components = {'continuous': [], 'categorical': []}
    best_val_loss = float('inf')
    train_cycle = 0
    val_cycle = 0
    
    # Initialize cycling variables
    current_subset_checkpoints = []
    current_subset_start_epoch = start_epoch
    
    # Initialize train and val subsets if starting from scratch
    if start_epoch == 0:
        if train_cycler:
            train_cycler.reset()  # Start from subset 0
        if validation_cycler:
            validation_cycler.reset()  # Start from subset 0
            
        # Create initial dataloaders with first subsets
        if train_cycler or validation_cycler:
            train_loader, val_loader = create_subset_dataloaders(
                train_df=train_df,
                val_df=val_df,
                train_tickers=train_cycler.get_current_subset() if train_cycler else None,
                val_tickers=validation_cycler.get_current_subset() if validation_cycler else None,
                config=config,
                debug=debug
            )

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize tracking variables
    best_models = []  # heap of (negative val_loss, epoch, cycle) tuples

    def calculate_loss(outputs, targets, loss_weights):
        """Calculate weighted combination of continuous and categorical losses."""
        if isinstance(targets, tuple):
            continuous_targets, categorical_targets = targets
        else:
            continuous_targets, categorical_targets = targets, None

        if isinstance(outputs, tuple):
            continuous_out, categorical_out = outputs
        else:
            continuous_out, categorical_out = outputs, None

        # MSE loss for continuous predictions
        continuous_loss = F.mse_loss(continuous_out, continuous_targets)

        # Cross entropy loss for each categorical feature
        categorical_loss = continuous_loss.new_tensor(0.0)
        if categorical_out is not None and categorical_targets is not None:
            for i in range(len(categorical_out)):
                categorical_loss += F.cross_entropy(categorical_out[i], categorical_targets[:, i])
            categorical_loss /= len(categorical_out)  # Average across features

        # Combine losses using weights
        total_loss = (loss_weights['continuous'] * continuous_loss +
                      loss_weights['categorical'] * categorical_loss)

        return total_loss, {
            'continuous': continuous_loss.item(),
            'categorical': categorical_loss.item()
        }

    def train_epoch(model, train_loader, optimizer, device, loss_weights, debug=False):
        model.train()
        total_loss = 0
        
        if debug:
            # Debug print DataLoader info
            print("\n=== Debug: Training Loop ===")
            print("DataLoader type:", type(train_loader))
            print("DataLoader length:", len(train_loader))
            print("DataLoader batch size:", train_loader.batch_size)
            print("DataLoader collate_fn:", train_loader.collate_fn)
        
        for batch_idx, batch in enumerate(train_loader):
            # Debug print first batch
            if batch_idx == 0 and debug:
                print("\nFirst batch info:")
                print("Batch type:", type(batch))
                print("Batch length:", len(batch))
                if isinstance(batch, (tuple, list)):
                    for i, item in enumerate(batch):
                        print(f"Item {i} type:", type(item))
                        if isinstance(item, torch.Tensor):
                            print(f"Item {i} shape:", item.shape)
                print("=== End Debug ===\n")
            
            # Unpack batch
            continuous_features, categorical_features, targets = batch
            
            # Move data to device
            continuous_features = continuous_features.to(device)
            if categorical_features is not None:
                categorical_features = categorical_features.to(device)
            if isinstance(targets, tuple):
                continuous_targets = targets[0].to(device)
                categorical_targets = targets[1].to(device) if targets[1] is not None else None
                targets = (continuous_targets, categorical_targets)
            else:
                targets = targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(continuous_features, categorical_features)
            
            # Calculate loss with both continuous and categorical components
            loss, loss_components = calculate_loss(outputs, targets, loss_weights)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            # Update weights
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            
            if debug and batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Total Loss = {loss.item():.6f}')
                print(f'  Continuous Loss = {loss_components["continuous"]:.6f}')
                print(f'  Categorical Loss = {loss_components["categorical"]:.6f}')
        
        return total_loss / len(train_loader)

    def validate(model, val_loader, device, loss_weights):
        model.eval()
        total_loss = 0
        loss_components_sum = {'continuous': 0, 'categorical': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                # Unpack batch
                continuous_features, categorical_features, targets = batch
                
                # Move data to device
                continuous_features = continuous_features.to(device)
                if categorical_features is not None:
                    categorical_features = categorical_features.to(device)
                if isinstance(targets, tuple):
                    continuous_targets = targets[0].to(device)
                    categorical_targets = targets[1].to(device) if targets[1] is not None else None
                    targets = (continuous_targets, categorical_targets)
                else:
                    targets = targets.to(device)
                
                # Forward pass
                outputs = model(continuous_features, categorical_features)
                
                # Calculate loss
                loss, batch_components = calculate_loss(outputs, targets, loss_weights)
                
                # Accumulate losses
                total_loss += loss.item()
                for k in loss_components_sum:
                    loss_components_sum[k] += batch_components[k]
        
        n_batches = len(val_loader)
        return (total_loss / n_batches, 
                {k: v / n_batches for k, v in loss_components_sum.items()})

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        if epoch >= max_epochs:
            print(f"Reached maximum epochs ({max_epochs})")
            break

        # Extract loss weights from config
        loss_weights = config['train_params'].get('loss_weights', {'continuous': 1.0, 'categorical': 1.0})

        avg_train_loss = train_epoch(model, train_loader, optimizer, device, loss_weights, debug)
        avg_val_loss, val_components = validate(model, val_loader, device, loss_weights)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Track component losses
        for k in val_components:
            val_loss_components[k].append(val_components[k])

        # Log training progress with component breakdowns
        print(f'Epoch {epoch:3d} | '
              f'Train Loss: {avg_train_loss:.6f} | '
              f'Val Loss: {avg_val_loss:.6f} | '
              f'Val Continuous: {val_components["continuous"]:.6f} | '
              f'Val Categorical: {val_components["categorical"]:.6f}')

        # Track the checkpoint for the current subset
        epochs_in_subset = epoch - current_subset_start_epoch + 1
        current_checkpoint = (avg_val_loss, epoch, None, epochs_in_subset)  # Placeholder for checkpoint name
        
        # Check if this is one of the N best checkpoints
        if len(current_subset_checkpoints) < train_params.get('checkpoints_per_subset', 3) or \
           avg_val_loss < max(c[0] for c in current_subset_checkpoints):
            
            # Create metadata
            metadata = TrainingMetadata(
                epoch=epoch,
                val_loss=avg_val_loss,
                train_loss=avg_train_loss,
                train_cycle=train_cycle,
                val_cycle=val_cycle,
                config=config,
                model_params={
                    'total_params': sum(p.numel() for p in model.parameters()),
                    'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                },
                training_history={
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_continuous_losses': val_loss_components['continuous'],
                    'val_categorical_losses': val_loss_components['categorical']
                },
                timestamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                trained_subsets=trained_subsets
            )
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'metadata': metadata
            }

            # Generate checkpoint name with cycle information
            cycle_info = f"_tc{train_cycler.current_subset_idx if train_cycler else 0}_vc{validation_cycler.current_subset_idx if validation_cycler else 0}"
            checkpoint_name = f"{config['train_params'].get('prefix', 'model')}_e{epoch}_valloss_{avg_val_loss:.7f}{cycle_info}.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            
            # Save checkpoint safely
            safe_torch_save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_name}")
            
            # Update current_checkpoint with the actual name
            current_checkpoint = (avg_val_loss, epoch, checkpoint_name, epochs_in_subset)
            
            # Add to current subset checkpoints, maintaining only the N best
            current_subset_checkpoints.append(current_checkpoint)
            current_subset_checkpoints.sort(key=lambda x: x[0])  # Sort by val_loss
            if len(current_subset_checkpoints) > train_params.get('checkpoints_per_subset', 3):
                # Remove the worst checkpoint file and entry
                _, _, worst_checkpoint_name, _ = current_subset_checkpoints.pop()
                worst_checkpoint_path = os.path.join(checkpoint_dir, worst_checkpoint_name)
                if os.path.exists(worst_checkpoint_path):
                    try:
                        os.remove(worst_checkpoint_path)
                        print(f"Removed checkpoint with higher loss: {worst_checkpoint_name}")
                    except OSError as e:
                        print(f"Error removing checkpoint {worst_checkpoint_name}: {e}")

        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
            print(f"Early stopping triggered after {epoch + 1} epochs")
            
            # Check if we have more train subsets
            if train_cycler and train_cycler.has_more_subsets():
                # Find best checkpoint from current subset
                valid_checkpoints = [(loss, ep, name) for loss, ep, name, epochs in current_subset_checkpoints if epochs >= min_epochs]
                
                if valid_checkpoints:
                    # Get the checkpoint with the lowest validation loss
                    best_loss, best_epoch, best_checkpoint_name = min(valid_checkpoints, key=lambda x: x[0])
                    print(f"\nRewinding to best checkpoint from current subset:")
                    print(f"Epoch: {best_epoch}, Val Loss: {best_loss:.7f}")
                    
                    # Load the best checkpoint
                    checkpoint = torch.load(os.path.join(checkpoint_dir, best_checkpoint_name))
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model = model.to(device)
                    
                    # Manage checkpoints for the completed subset
                    manage_checkpoints(
                        checkpoint_dir=checkpoint_dir,
                        current_subset_checkpoints=current_subset_checkpoints,
                        best_models=best_models,
                        min_epochs_per_subset=min_epochs,
                        current_cycle=train_cycle
                    )
                
                # Reset checkpoint tracking for new subset
                current_subset_checkpoints = []
                current_subset_start_epoch = epoch + 1
                
                # Reset early stopping for new subset
                early_stopping = EarlyStopping(
                    patience=patience,
                    min_epochs=min_epochs,
                    min_delta=train_params.get('min_delta', 0)
                )
                
                # Move to next training subset
                print(f"\nSwitching to next training subset after epoch {epoch}")
                next_train_subset = train_cycler.next_subset()
                train_cycle += 1
                trained_subsets.append(next_train_subset)
                
                # Create new dataloaders with updated subsets
                train_loader, val_loader = create_subset_dataloaders(
                    train_df=train_df,
                    val_df=val_df,
                    train_tickers=next_train_subset,
                    val_tickers=validation_cycler.get_current_subset() if validation_cycler else None,
                    config=config,
                    debug=debug
                )
                
            else:
                # If we have no more train subsets but still have val subsets, end training
                if validation_cycler and validation_cycler.has_more_subsets():
                    print("\nNo more training subsets available. Ending training.")
                break

    return train_losses, val_losses

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

def manage_checkpoints(checkpoint_dir: str, current_subset_checkpoints: List[Tuple[float, int, str, int]], 
                      best_models: List[Tuple[float, int, Tuple[int, int]]], min_epochs_per_subset: int = 2, 
                      current_cycle: int = 0) -> None:
    """Manage checkpoints when transitioning between subsets.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        current_subset_checkpoints: List of (val_loss, epoch, checkpoint_name, epochs_in_subset)
        best_models: Global list of best models (val_loss, epoch, cycle)
        min_epochs_per_subset: Minimum number of epochs a model should train on a subset
        current_cycle: Current training cycle number (increments each time we cycle through all subsets)
    """
    if not current_subset_checkpoints:
        return
    
    # Find the best checkpoint that has trained for at least min_epochs_per_subset
    qualified_checkpoints = [(loss, epoch, name, epochs) for loss, epoch, name, epochs 
                           in current_subset_checkpoints if epochs >= min_epochs_per_subset]
    
    if qualified_checkpoints:
        # Keep only the best qualified checkpoint from this subset
        best_checkpoint = min(qualified_checkpoints, key=lambda x: x[0])
        best_loss, best_epoch, best_name, _ = best_checkpoint
        
        # Add to global best models list with cycle information
        heapq.heappush(best_models, (best_loss, best_epoch, (current_cycle, best_epoch)))
        
        # Delete all other checkpoints from this subset
        for loss, epoch, name, _ in current_subset_checkpoints:
            if name != best_name:  # Don't delete the best checkpoint
                try:
                    checkpoint_path = os.path.join(checkpoint_dir, name)
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                        print(f"Removed non-best checkpoint from subset: {name}")
                except OSError as e:
                    print(f"Error removing checkpoint {name}: {e}")
        
        # Enforce global maximum checkpoint constraint
        while len(best_models) > 10:
            # Get the worst model from our heap of best models
            worst_loss, worst_epoch, (worst_cycle, _) = heapq.heappop(best_models)
            
            # Find and delete its checkpoint
            for f in glob.glob(os.path.join(checkpoint_dir, "*.pt")):
                try:
                    checkpoint = torch.load(f, map_location='cpu')
                    if (checkpoint['metadata'].train_cycle == worst_cycle and 
                        checkpoint['epoch'] == worst_epoch):
                        os.remove(f)
                        print(f"Removed checkpoint to maintain global maximum: {os.path.basename(f)}")
                        break
                except Exception as e:
                    print(f"Error processing checkpoint {f}: {e}")

def safe_torch_save(checkpoint, save_path, max_retries=3):
    """Safely save PyTorch checkpoint with retries and temporary file."""
    save_dir = os.path.dirname(save_path)
    
    # Check available disk space (need roughly 2x the checkpoint size for safe saving)
    checkpoint_size = sum(param.numel() * param.element_size() 
                         for param in checkpoint['model_state_dict'].values())
    free_space = shutil.disk_usage(save_dir).free
    
    if free_space < checkpoint_size * 3:  # 3x safety factor
        raise RuntimeError(f"Insufficient disk space. Need {checkpoint_size*3} bytes, have {free_space}")
    
    # Create temporary file
    temp_save_path = None
    for attempt in range(max_retries):
        try:
            # Save to temporary file first
            with tempfile.NamedTemporaryFile(delete=False, dir=save_dir) as tmp_file:
                temp_save_path = tmp_file.name
                torch.save(checkpoint, temp_save_path)
            
            # If save was successful, move the temp file to the final location
            shutil.move(temp_save_path, save_path)
            return True
            
        except (RuntimeError, OSError) as e:
            print(f"Save attempt {attempt + 1} failed: {str(e)}")
            if temp_save_path and os.path.exists(temp_save_path):
                try:
                    os.remove(temp_save_path)
                except OSError:
                    pass
            
            if attempt == max_retries - 1:
                print(f"Failed to save checkpoint after {max_retries} attempts")
                raise
            
            # Wait before retry (exponential backoff)
            time.sleep(2 ** attempt)
    
    return False

def signal_handler(signum, frame):
    global interrupted
    print("\nInterrupt received. Will save checkpoint and exit after current batch...")
    interrupted = True

interrupted = False
