# flake8: noqa E501
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from torch.utils.data import DataLoader
import os
from heapq import heappush, heappushpop
from early_stopping import EarlyStopping
from ticker_cycler import TickerCycler
import random
from data_loading import create_subset_dataloaders
import pandas as pd
import signal
import sys
from dataclasses import dataclass
from torch.serialization import add_safe_globals
from optimizers import CAdamW
import time
import shutil
from pathlib import Path
import tempfile
import glob

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
        config: Configuration dictionary
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
        min_delta=train_params.get('min_delta', 0),
        max_checkpoints=train_params.get('max_checkpoints', 3)
    )

    # Initialize training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    train_cycle = 0
    val_cycle = 0
    
    # Track cycle completion status
    train_cycle_completed = not train_cycler  # True if train cycling is disabled
    val_cycle_completed = not validation_cycler  # True if validation cycling is disabled

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize tracking variables
    current_subset_checkpoints = []
    current_subset_start_epoch = 0
    best_models = []  # heap of (negative val_loss, epoch, cycle) tuples

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        if epoch >= max_epochs:
            print(f"Reached maximum epochs ({max_epochs})")
            break

        model.train()
        total_train_loss = 0
        num_batches = 0

        # Training phase
        for batch_idx, (features, targets) in enumerate(train_loader):
            # Unpack features into continuous and categorical
            continuous_features, categorical_features = features
            
            # Move to device
            continuous_features = continuous_features.to(device)
            categorical_features = categorical_features.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(continuous_features, categorical_features)
            loss = F.mse_loss(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            num_batches += 1

            if debug and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")

        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for features, targets in val_loader:
                # Unpack features into continuous and categorical
                continuous_features, categorical_features = features
                
                # Move to device
                continuous_features = continuous_features.to(device)
                categorical_features = categorical_features.to(device)
                targets = targets.to(device)
                
                outputs = model(continuous_features, categorical_features)
                val_loss = F.mse_loss(outputs, targets)
                
                total_val_loss += val_loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)

        # Print epoch results
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
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
                    'val_losses': val_losses
                },
                timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
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
            
            # Track checkpoint for current subset
            epochs_in_subset = epoch - current_subset_start_epoch + 1
            current_subset_checkpoints.append((avg_val_loss, epoch, checkpoint_name, epochs_in_subset))
            
            # Save checkpoint safely
            safe_torch_save(checkpoint, checkpoint_path)

        # Early stopping check
        if early_stopping(avg_val_loss, epoch):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            
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
                    
                    # Reinitialize optimizer and scheduler
                    optimizer = OPTIMIZER(optimizer_grouped_parameters, lr=learning_rate)
                    scheduler = get_scheduler(optimizer, config, len(train_loader))
                
                # Reset checkpoint tracking for new subset
                current_subset_checkpoints = []
                current_subset_start_epoch = epoch + 1
                
                # Move to next training subset
                print(f"\nSwitching to next training subset after epoch {epoch}")
                train_cycler.next_subset()
                train_cycle += 1
                
                # Update validation subset if cycling is enabled
                if validation_cycler:
                    if not validation_cycler.has_more_subsets():
                        validation_cycler.reset()
                        val_cycle_completed = True
                    validation_cycler.next_subset()
                    val_cycle += 1
                
                # Create new dataloaders with updated subsets
                train_loader = create_subset_dataloaders(
                    df=train_df,
                    config=config,
                    tickers=train_cycler.get_current_subset(),
                    batch_size=config['train_params']['batch_size'],
                    sequence_length=config['data_params']['sequence_length'],
                    debug=debug
                )
                
                if validation_cycler:
                    val_loader = create_subset_dataloaders(
                        df=val_df,
                        config=config,
                        tickers=validation_cycler.get_current_subset(),
                        batch_size=config['train_params']['batch_size'],
                        sequence_length=config['data_params']['sequence_length'],
                        debug=debug
                    )
            else:
                break

        # Check if we should cycle validation set
        if validation_cycler and validation_cycler.should_cycle(epoch):
            print("\nCycling validation set...")
            val_cycle += 1
            val_loader = create_subset_dataloaders(
                df=val_df,
                config=config,
                tickers=validation_cycler.get_next_subset(),
                batch_size=config['train_params']['batch_size'],
                sequence_length=config['data_params']['sequence_length'],
                debug=debug
            )

        # Check if we should cycle training set
        if train_cycler and train_cycler.should_cycle(epoch):
            print("\nCycling training set...")
            train_cycle += 1
            next_subset = train_cycler.get_next_subset()
            trained_subsets.append(next_subset)
            train_loader = create_subset_dataloaders(
                df=train_df,
                config=config,
                tickers=next_subset,
                batch_size=config['train_params']['batch_size'],
                sequence_length=config['data_params']['sequence_length'],
                debug=debug
            )

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
                      best_models: List[Tuple[float, int, Tuple[int, int]]], max_checkpoints: int,
                      min_epochs_per_subset: int = 2, current_cycle: int = 0) -> None:
    """Manage checkpoints when transitioning between subsets.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        current_subset_checkpoints: List of (val_loss, epoch, checkpoint_name, epochs_in_subset)
        best_models: Global list of best models (val_loss, epoch, cycle)
        max_checkpoints: Maximum number of checkpoints to keep globally
        min_epochs_per_subset: Minimum number of epochs a model should train on a subset
        current_cycle: Current training cycle number (increments each time we cycle through all subsets)
    """
    if not current_subset_checkpoints:
        return
    
    # Find the best checkpoint that has trained for at least min_epochs_per_subset
    qualified_checkpoints = [(loss, epoch, name, epochs) for loss, epoch, name, epochs 
                           in current_subset_checkpoints if epochs >= min_epochs_per_subset]
    
    if qualified_checkpoints:
        # Keep only the best qualified checkpoint from this cycle
        best_checkpoint = min(qualified_checkpoints, key=lambda x: x[0])
        keep_file = best_checkpoint[2]
        
        # Get all checkpoints for the current subset (based on subset identifier in filename)
        subset_identifier = keep_file.split('_')[1]  # Assuming format like "checkpoint_subset1_epoch10.pt"
        all_subset_checkpoints = []
        for f in glob.glob(os.path.join(checkpoint_dir, f"*_{subset_identifier}_*.pt")):
            try:
                # Extract cycle number from metadata in checkpoint
                checkpoint = torch.load(f, map_location='cpu')
                checkpoint_cycle = checkpoint['metadata'].train_cycle
                all_subset_checkpoints.append((f, checkpoint_cycle))
            except Exception as e:
                print(f"Error reading checkpoint {f}: {e}")
                continue
        
        # Keep only the checkpoint from the current cycle
        for checkpoint_path, cycle_num in all_subset_checkpoints:
            checkpoint_name = os.path.basename(checkpoint_path)
            if checkpoint_name != keep_file:  # Don't delete our current best checkpoint
                try:
                    os.remove(checkpoint_path)
                    print(f"Deleted old checkpoint from cycle {cycle_num}: {checkpoint_name}")
                except OSError as e:
                    print(f"Error deleting checkpoint {checkpoint_name}: {e}")
        
        # Ensure we don't exceed max_checkpoints globally
        # Only consider checkpoints with the same prefix (from current config)
        prefix = keep_file.split('_')[0]  # Extract prefix from current checkpoint name
        all_checkpoints = glob.glob(os.path.join(checkpoint_dir, f"{prefix}_*.pt"))
        if len(all_checkpoints) > max_checkpoints:
            # Sort checkpoints by modification time (oldest first)
            checkpoints_with_time = [(f, os.path.getmtime(f)) for f in all_checkpoints]
            checkpoints_with_time.sort(key=lambda x: x[1])
            
            # Delete oldest checkpoints until we're at max_checkpoints
            for checkpoint_path, _ in checkpoints_with_time[:-max_checkpoints]:
                try:
                    os.remove(checkpoint_path)
                    print(f"Deleted old checkpoint: {os.path.basename(checkpoint_path)}")
                except OSError as e:
                    print(f"Error deleting old checkpoint {checkpoint_path}: {e}")

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
