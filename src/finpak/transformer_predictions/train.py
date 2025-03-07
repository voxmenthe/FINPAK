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
import statistics

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

def get_cycles_completed(scheduler_config: Dict[str, Any], current_step: int, steps_per_epoch: int) -> int:
    """
    Calculate how many full learning rate cycles have completed.
    
    Args:
        scheduler_config: The scheduler configuration dictionary
        current_step: Current training step
        steps_per_epoch: Number of steps per epoch
        
    Returns:
        Number of completed learning rate cycles
    """
    scheduler_type = scheduler_config['type']
    
    if scheduler_type == "cyclical":
        cycle_length = scheduler_config['cycle_params']['cycle_length'] * steps_per_epoch
        return current_step // cycle_length
        
    elif scheduler_type == "one_cycle":
        # one_cycle only has one cycle total
        return 1 if current_step >= steps_per_epoch * scheduler_config['warmup_epochs'] else 0
        
    elif scheduler_type == "warmup_decay":
        # warmup_decay doesn't have cycles, so we'll use epochs as a proxy
        warmup_steps = scheduler_config['warmup_epochs'] * steps_per_epoch
        total_steps = steps_per_epoch * scheduler_config.get('total_epochs', 0)
        if total_steps == 0:
            return 0
        # Consider each quarter of the training duration after warmup as a "cycle"
        steps_after_warmup = max(0, current_step - warmup_steps)
        remaining_steps = total_steps - warmup_steps
        return int(4 * steps_after_warmup / remaining_steps) if remaining_steps > 0 else 0
    
    return 0

def get_cycle_length(scheduler_config: Dict[str, Any]) -> int:
    """
    Get the cycle length in epochs based on scheduler configuration.
    
    Args:
        scheduler_config: The scheduler configuration dictionary
        
    Returns:
        Number of epochs per cycle
    """
    scheduler_type = scheduler_config['type']
    
    if scheduler_type == "cyclical":
        return scheduler_config['cycle_params']['cycle_length']
    elif scheduler_type == "one_cycle":
        return scheduler_config['warmup_epochs'] * 2  # warmup + cooldown
    elif scheduler_type == "warmup_decay":
        # For warmup_decay, consider the period after warmup as one cycle
        return scheduler_config['warmup_epochs'] * 2
    
    return 1  # Default to 1 if unknown scheduler type

def get_checkpoint_cycle_info(checkpoint_path: str, scheduler_config: Dict[str, Any], steps_per_epoch: int) -> Tuple[int, int]:
    """
    Get the learning rate cycle information from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        scheduler_config: Scheduler configuration
        steps_per_epoch: Number of steps per epoch
        
    Returns:
        Tuple of (epoch, cycles_completed)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint['metadata'].epoch
    current_step = epoch * steps_per_epoch
    cycles_completed = get_cycles_completed(scheduler_config, current_step, steps_per_epoch)
    return epoch, cycles_completed

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
    trained_subsets: Optional[List[List[str]]] = None,
    metrics_df: Optional[pd.DataFrame] = None
) -> Tuple[List[float], List[float], pd.DataFrame]:
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
        metrics_df: DataFrame to store training metrics
    """

    if config is None:
        raise ValueError("Config must be provided")

    # Get cycle length and calculate derived parameters
    cycle_length = get_cycle_length(config['train_params']['scheduler'])
    min_epochs_per_subset = config['train_params'].get('min_epochs_per_subset', cycle_length * 2)
    checkpoints_per_cycle = min(int(cycle_length / 3), min_epochs_per_subset)
    
    # Initialize early stopping with dynamic patience based on cycle length
    base_patience = config['train_params']['patience']
    cycle_based_patience = (cycle_length * 2) + base_patience
    
    early_stop = EarlyStopping(
        patience=cycle_based_patience,
        min_delta=config['train_params']['min_delta'],
        max_checkpoints=checkpoints_per_cycle,
        min_epochs=min_epochs_per_subset
    )
    
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
    
    # Track cycle completion status
    train_cycle_completed = not enable_train_cycling  # True if train cycling is disabled
    val_cycle_completed = not enable_validation_cycling  # True if validation cycling is disabled
    
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

    # Initialize heap for keeping track of best models (max heap using negative loss)
    best_models = []
    
    # Keep track of checkpoints and validation losses for current training subset
    current_subset_checkpoints = []  # List of (val_loss, epoch, checkpoint_name, epochs_in_subset)
    current_subset_start_epoch = 0
    
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
    optimizer = OPTIMIZER(
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
    
    # Set up interrupt handling
    interrupted = False
    def signal_handler(signum, frame):
        nonlocal interrupted
        print("\nInterrupt received. Will save checkpoint and exit after current batch...")
        interrupted = True
    
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
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

    # Initialize tracking variables
    current_subset_checkpoints = []
    current_subset_start_epoch = 0
    best_models = []  # heap of (negative val_loss, epoch, cycle) tuples
    train_cycle = 0  # Track how many times we've cycled through all subsets
    
    for epoch in range(start_epoch, n_epochs):
        # Get current subsets
        current_train_subset = train_cycler.get_current_subset()
        current_val_subset = validation_cycler.get_current_subset()
        
        # Training loop
        model.train()
        epoch_train_loss = 0.0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            if interrupted:
                print("Saving final checkpoint before exit...")
                metadata = TrainingMetadata(
                    epoch=epoch,
                    val_loss=val_losses[-1] if val_losses else float('inf'),
                    train_loss=train_losses[-1] if train_losses else float('inf'),
                    train_cycle=train_cycler.current_subset_idx if train_cycler else 0,
                    val_cycle=validation_cycler.current_subset_idx if validation_cycler else 0,
                    config=config,
                    model_params={
                        'total_params': sum(p.numel() for p in model.parameters()),
                        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                    },
                    training_history={
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                    },
                    timestamp=pd.Timestamp.now().isoformat(),
                    trained_subsets=trained_subsets + [train_cycler.get_current_subset()] if train_cycler else []
                )
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metadata': metadata
                }
                
                final_checkpoint_name = f"{config['train_params']['prefix']}_id_{config['train_params']['run_id']}_INTERRUPTED_e{epoch}.pt"
                torch.save(checkpoint, os.path.join(checkpoint_dir, final_checkpoint_name))
                print(f"Saved interrupt checkpoint: {final_checkpoint_name}")
                sys.exit(0)
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Debug prints for input data
            if debug:
                print("\n=== Training Batch Debug ===")
                print(f"Input stats: min={batch_x.min().item():.4f}, max={batch_x.max().item():.4f}")
                print(f"Target stats: min={batch_y.min().item():.4f}, max={batch_y.max().item():.4f}")
                print(f"NaN in input: {torch.isnan(batch_x).sum().item()}")
                print(f"NaN in target: {torch.isnan(batch_y).sum().item()}")
            
            predictions = model(batch_x)
            loss = F.mse_loss(predictions, batch_y)
            
            # Debug prints for predictions and loss
            if debug:
                print(f"Pred stats: min={predictions.min().item():.4f}, max={predictions.max().item():.4f}")
                print(f"NaN in predictions: {torch.isnan(predictions).sum().item()}")
                print(f"Loss: {loss.item():.4f}")
                if torch.isnan(loss):
                    print("WARNING: NaN loss detected!")
                    print("Last learning rate:", scheduler.get_last_lr()[0])
            
            loss.backward()
            
            # # Debug prints for gradients
            # if debug:
            #     for name, param in model.named_parameters():
            #         if param.grad is not None:
            #             grad_norm = param.grad.norm().item()
            #             if torch.isnan(param.grad).any():
            #                 print(f"NaN gradient in {name}")
            #             if grad_norm > 1.0:
            #                 print(f"Large gradient in {name}: {grad_norm:.4f}")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_train_loss += loss.item()
            
            if debug and epoch == 0 and batch_idx == 0:
                print("\nFirst Batch Statistics:")
                print(f"Batch size: {batch_x.size(0)}")
                print(f"Expected batch size: {config['train_params']['batch_size']}")
                print(f"Total training samples: {len(train_loader.dataset)}")
                if config.get('augmentation_params', {}).get('enabled', False):
                    aug_fraction = config['augmentation_params']['subset_fraction']
                    expected_aug_samples = int(len(train_loader.dataset) / (1 + aug_fraction))
                    print(f"Estimated original samples: {expected_aug_samples}")
                    print(f"Estimated augmented samples: {len(train_loader.dataset) - expected_aug_samples}")
            
        # Calculate average losses
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation loop
        model.eval()
        epoch_val_loss = 0.0  # Initialize the validation loss accumulator
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                predictions = model(batch_x)
                epoch_val_loss += F.mse_loss(predictions, batch_y).item()  # Use epoch_val_loss instead of val_loss
                
        current_val_loss = epoch_val_loss / len(val_loader)  # Average the loss over all batches
        val_losses.append(current_val_loss)
        
        # Initialize should_continue before early stopping check
        should_continue = True
        
        # Save checkpoint if it's among the best
        metadata = TrainingMetadata(
            epoch=epoch,
            val_loss=current_val_loss,
            train_loss=train_losses[-1],
            train_cycle=train_cycler.current_subset_idx if train_cycler else 0,
            val_cycle=validation_cycler.current_subset_idx if validation_cycler else 0,
            config=config,
            model_params={
                'total_params': sum(p.numel() for p in model.parameters()),
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
            training_history={
                'train_losses': train_losses,
                'val_losses': val_losses,
            },
            timestamp=pd.Timestamp.now().isoformat(),
            trained_subsets=trained_subsets + [train_cycler.get_current_subset()] if train_cycler else []
        )
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metadata': metadata
        }
        
        # Generate checkpoint name with cycle information
        cycle_info = f"_tc{train_cycler.current_subset_idx if train_cycler else 0}_vc{validation_cycler.current_subset_idx if validation_cycler else 0}"
        checkpoint_name = f"{config['train_params']['prefix']}_id_{config['train_params']['run_id']}_arc_{config['train_params']['architecture_version']}{cycle_info}_e{epoch}_valloss_{current_val_loss:.7f}.pt"
        
        # Track checkpoint for current subset
        epochs_in_subset = epoch - current_subset_start_epoch + 1
        current_subset_checkpoints.append((current_val_loss, epoch, checkpoint_name, epochs_in_subset))
        
        # Always save the first checkpoint of each cycle
        current_cycle = (train_cycler.current_subset_idx if train_cycler else 0, validation_cycler.current_subset_idx if validation_cycler else 0)
        is_first_checkpoint_in_cycle = not any(cycle == current_cycle for _, _, cycle in best_models)
        
        # Handle model checkpointing
        if len(best_models) < train_params['max_checkpoints'] or is_first_checkpoint_in_cycle:
            heappush(best_models, (-current_val_loss, epoch, current_cycle))
            save_path = os.path.join(checkpoint_dir, checkpoint_name)
            if safe_torch_save(checkpoint, save_path):
                print(f"Saved chkpt cycle# {current_cycle}  {checkpoint_name}")
        else:
            # If current model is better than worst of our best models
            if -current_val_loss > best_models[0][0]:
                heappush(best_models, (-current_val_loss, epoch, current_cycle))
                save_path = os.path.join(checkpoint_dir, checkpoint_name)
                if safe_torch_save(checkpoint, save_path):
                    print(f"Saved chkpt cycle# {current_cycle}  {checkpoint_name}")

        # Manage checkpoints when switching to a new subset
        if train_cycler and not train_cycler.has_more_subsets():
            manage_checkpoints(checkpoint_dir, current_subset_checkpoints, best_models, 
                             min_epochs_per_subset=min_epochs_per_subset, max_checkpoints=train_params['max_checkpoints'],
                             current_cycle=train_cycle)
            current_subset_checkpoints = []
            current_subset_start_epoch = epoch + 1
            
            # Increment cycle counter when we've gone through all subsets
            train_cycle += 1
        
        # Check early stopping
        current_step = epoch * len(train_loader) + len(train_loader) - 1
        cycles_completed = get_cycles_completed(
            config['train_params']['scheduler'], 
            current_step, 
            len(train_loader)
        )
        
        epochs_in_subset = epoch - current_subset_start_epoch + 1
        min_cycles_complete = cycles_completed >= 2
        min_epochs_complete = epochs_in_subset >= min_epochs_per_subset
        
        # Save checkpoint if it's the best so far for this subset
        if early_stop.is_best(current_val_loss):
            metadata = TrainingMetadata(
                epoch=epoch,
                val_loss=current_val_loss,
                train_loss=train_losses[-1],
                train_cycle=train_cycler.current_subset_idx if train_cycler else 0,
                val_cycle=validation_cycler.current_subset_idx if validation_cycler else 0,
                config=config,
                model_params={
                    'total_params': sum(p.numel() for p in model.parameters()),
                    'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                },
                training_history={
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                },
                timestamp=pd.Timestamp.now().isoformat(),
                trained_subsets=trained_subsets + [train_cycler.get_current_subset()] if train_cycler else []
            )
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metadata': metadata
            }
            
            checkpoint_name = (
                f"{config['train_params']['prefix']}_"
                f"id_{config['train_params']['run_id']}_"
                f"arc_{config['train_params']['architecture_version']}_"
                f"tc{train_cycler.current_subset_idx if train_cycler else 0}_"
                f"vc{validation_cycler.current_subset_idx if validation_cycler else 0}_"
                f"e{epoch}_"
                f"valloss_{current_val_loss:.7f}.pt"
            )
            
            current_subset_checkpoints.append((current_val_loss, epoch, checkpoint_name, epochs_in_subset))
            
            save_path = os.path.join(checkpoint_dir, checkpoint_name)
            if safe_torch_save(checkpoint, save_path):
                print(f"Saved best checkpoint for current subset: {checkpoint_name}")
        
        # Only check early stopping if minimum requirements are met
        if min_cycles_complete and min_epochs_complete:
            if early_stop(current_val_loss):
                print(f"Early stopping triggered for current subset after {cycles_completed} cycles")
                
                if train_cycler and train_cycler.has_more_subsets():
                    # Filter checkpoints to only those after first learning rate cycle
                    valid_checkpoints = []
                    for loss, ep, name, epochs in current_subset_checkpoints:
                        # Only consider checkpoints that were among the best (i.e., actually saved)
                        checkpoint_path = os.path.join(checkpoint_dir, name)
                        if (epochs >= min_epochs_per_subset and 
                            os.path.exists(checkpoint_path) and 
                            loss <= statistics.median([x[0] for x in current_subset_checkpoints])):  # Only consider better than median performance
                            try:
                                checkpoint_epoch, cycles = get_checkpoint_cycle_info(
                                    checkpoint_path,
                                    config['train_params']['scheduler'],
                                    len(train_loader)
                                )
                                if cycles >= 1:  # Only consider checkpoints after first cycle
                                    valid_checkpoints.append((loss, checkpoint_epoch, name))
                            except Exception as e:
                                print(f"Warning: Error reading checkpoint {name}: {e}")
                                continue
                    
                    if valid_checkpoints:
                        # Load best checkpoint after first cycle
                        best_loss, best_epoch, best_name = min(valid_checkpoints, key=lambda x: x[0])
                        print(f"Loading best checkpoint for next subset: Epoch {best_epoch}, Loss {best_loss:.7f}")
                        
                        checkpoint = torch.load(os.path.join(checkpoint_dir, best_name))
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model = model.to(device)
                        
                        # Move to next subset
                        train_cycler.next_subset()
                        
                        # Calculate effective training progress
                        epochs_trained_in_subset = epoch - current_subset_start_epoch
                        total_epochs_trained = sum(1 for cp in current_subset_checkpoints if cp[3] >= min_epochs_per_subset)
                        
                        # Adjust scheduler steps based on effective training progress
                        effective_steps = total_epochs_trained * len(train_loader)
                        
                        # Reset optimizer and scheduler with appropriate state
                        optimizer = OPTIMIZER(
                            model.parameters(),
                            lr=config['train_params']['scheduler']['base_lr'],
                            betas=(0.9, 0.95),
                            weight_decay=weight_decay
                        )
                        
                        scheduler = get_scheduler(optimizer, config, len(train_loader))
                        
                        # Fast-forward scheduler to appropriate point
                        for _ in range(effective_steps):
                            scheduler.step()
                        
                        # Update tracking variables
                        current_subset_start_epoch = epoch + 1
                        current_subset_checkpoints = []
                        
                        # Reset early stopping for new subset
                        early_stop = EarlyStopping(
                            patience=cycle_based_patience,
                            min_delta=min_delta,
                            max_checkpoints=checkpoints_per_cycle,
                            min_epochs=min_epochs_per_subset
                        )
                        
                        # Update validation cycler if needed
                        if enable_validation_cycling:
                            if not validation_cycler.has_more_subsets():
                                validation_cycler.reset()
                            validation_cycler.next_subset()
                        
                        # Create new dataloaders for next subset
                        train_loader, val_loader = create_subset_dataloaders(
                            train_df=train_df,
                            val_df=val_df,
                            train_tickers=train_cycler.get_current_subset(),
                            val_tickers=validation_cycler.get_current_subset() if enable_validation_cycling else None,
                            config=config,
                            debug=debug
                        )
                        
                        # Reinitialize optimizer with loaded model parameters
                        optimizer = OPTIMIZER(
                            model.parameters(),
                            lr=config['train_params']['scheduler']['base_lr'],
                            betas=(0.9, 0.95),
                            weight_decay=weight_decay
                        )
                        
                        # Reinitialize scheduler
                        scheduler = get_scheduler(optimizer, config, len(train_loader))
                        
                        continue
                else:
                    # All subsets completed
                    print("Training completed - all subsets processed")
                    break

        # Only reset early stopping flags if we're actually continuing
        # This prevents resetting when we should be stopping
        if not should_continue:
            print(f"\nEarly stopping triggered after epoch {epoch}")
            # Save final checkpoint
            metadata = TrainingMetadata(
                epoch=epoch,
                val_loss=current_val_loss,
                train_loss=train_losses[-1],
                train_cycle=train_cycler.current_subset_idx if train_cycler else 0,
                val_cycle=validation_cycler.current_subset_idx if validation_cycler else 0,
                config=config,
                model_params={
                    'total_params': sum(p.numel() for p in model.parameters()),
                    'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                },
                training_history={
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                },
                timestamp=pd.Timestamp.now().isoformat(),
                trained_subsets=trained_subsets + [train_cycler.get_current_subset()] if train_cycler else []
            )
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metadata': metadata
            }
            final_checkpoint_name = (
                f"{config['train_params']['prefix']}_"
                f"final_id_{config['train_params']['run_id']}_"
                f"arc_{config['train_params']['architecture_version']}_"
                f"tc{train_cycler.current_subset_idx if train_cycler else 0}_"
                f"vc{validation_cycler.current_subset_idx if validation_cycler else 0}_"
                f"e{epoch}_"
                f"valloss_{current_val_loss:.7f}.pt"
            )
            torch.save(checkpoint, os.path.join(checkpoint_dir, final_checkpoint_name))
            print(f"Saved final chkpt: {final_checkpoint_name}")
            break

        # Add metrics to DataFrame
        new_row = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': current_val_loss,
            'train_subset_tickers': ','.join(current_train_subset),
            'test_subset_tickers': ','.join(current_val_subset)
        }


        if (epoch + 1) % train_params['print_every'] == 0:
            current_step = epoch * len(train_loader)  # Calculate current step
            print(f"Epoch: {epoch+1}/{n_epochs} (Step: {current_step})")
            print(f"Training loss: {train_losses[-1]:.7f}")
            print(f"Validation loss: {val_losses[-1]:.7f}")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
            metrics_fname = f"{config['train_params']['prefix']}_id_{config['train_params']['run_id']}_e{epoch}_metrics.csv"
            metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)
            metrics_df.to_csv(metrics_fname, index=False)


    # At the end of training, print information about best checkpoints
    print("\nBest checkpoints:")
    for neg_loss, epoch, cycle in sorted(best_models, reverse=True):
        print(f"Epoch {epoch}: validation loss = {-neg_loss:.7f}")
            
    return train_losses, val_losses, metrics_df
