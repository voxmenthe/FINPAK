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

    # Get cycle length for early stopping configuration
    cycle_length = get_cycle_length(config['train_params']['scheduler'])
    
    # Initialize early stopping with dynamic patience based on cycle length
    base_patience = config['train_params']['patience']
    cycle_based_patience = (cycle_length * 2) + base_patience
    
    # Get checkpoint rewinding parameters
    rewind_quantile_divisions = config['train_params'].get('rewind_quantile_divisions', 10)  # Default to deciles
    rewind_min_extra_epochs = config['train_params'].get('rewind_min_extra_epochs', 5)  # Minimum extra epochs to prefer longer-trained checkpoint
    
    early_stop = EarlyStopping(
        patience=cycle_based_patience,  # Use cycle-based patience instead of base patience
        min_delta=min_delta,
        max_checkpoints=train_params['max_checkpoints'],
        min_epochs=max(min_epochs_before_stopping, cycle_length * 2)  # Ensure minimum epochs is at least 2 cycles
    )
    
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

    def get_value_quantile(value: float, values: List[float], n_quantiles: int = 10) -> int:
        """
        Determine which quantile a value falls into within a distribution.
        
        Args:
            value: The value to check
            values: List of all values in the distribution
            n_quantiles: Number of quantiles to divide the distribution into
            
        Returns:
            The quantile index (0 to n_quantiles-1) that the value falls into
        """
        if not values:
            return 0
        
        # Calculate quantile boundaries
        boundaries = [np.quantile(values, q) for q in np.linspace(0, 1, n_quantiles+1)]
        
        # Find which quantile the value falls into
        for i in range(len(boundaries)-1):
            if boundaries[i] <= value <= boundaries[i+1]:
                return i
            
        return n_quantiles-1  # For any values above the highest boundary

    def select_best_checkpoint(valid_checkpoints: List[Tuple[float, int, str]], 
                             rewind_quantile_divisions: int,
                             rewind_min_extra_epochs: int) -> Tuple[float, int, str]:
        """
        Select the best checkpoint considering both validation loss and training duration.
        
        Args:
            valid_checkpoints: List of tuples (val_loss, epoch, checkpoint_name)
            rewind_quantile_divisions: Number of quantiles to divide validation losses into
            rewind_min_extra_epochs: Minimum extra epochs required to prefer a "good enough" checkpoint
            
        Returns:
            The selected checkpoint tuple (val_loss, epoch, checkpoint_name)
        """
        if not valid_checkpoints:
            raise ValueError("No valid checkpoints provided")
        
        # Sort by validation loss to find the best one
        valid_checkpoints.sort(key=lambda x: x[0])
        best_checkpoint = valid_checkpoints[0]
        best_loss, best_epoch, _ = best_checkpoint
        
        # Get all validation losses
        all_losses = [x[0] for x in valid_checkpoints]
        
        # Find the quantile of the best loss
        best_quantile = get_value_quantile(best_loss, all_losses, rewind_quantile_divisions)
        
        # Find checkpoints in the same quantile that have trained longer
        same_quantile_checkpoints = [
            cp for cp in valid_checkpoints 
            if get_value_quantile(cp[0], all_losses, rewind_quantile_divisions) == best_quantile
            and cp[1] > best_epoch + rewind_min_extra_epochs
        ]
        
        if same_quantile_checkpoints:
            # Among checkpoints in the same quantile that have trained longer,
            # choose the one with the best validation loss
            return min(same_quantile_checkpoints, key=lambda x: x[0])
        
        return best_checkpoint

    # Initialize tracking variables
    current_subset_checkpoints = []
    current_subset_start_epoch = 0
    best_models = []  # heap of (negative val_loss, epoch, cycle) tuples
    train_cycle = 0  # Track how many times we've cycled through all subsets
    
    for epoch in range(start_epoch, n_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        
        # Check if we need to cycle train/val datasets
        if train_cycler and not train_cycler.has_more_subsets():
            print(f"\nResetting train cycler at epoch {epoch}")
            train_cycler.reset_and_randomize()
            train_cycle += 1
            train_cycler.next_subset()
            
            # Manage checkpoints for the completed cycle
            manage_checkpoints(checkpoint_dir, current_subset_checkpoints, best_models, 
                             train_params['max_checkpoints'], current_cycle=train_cycle)
            current_subset_checkpoints = []
            current_subset_start_epoch = epoch
        
        if validation_cycler and not validation_cycler.has_more_subsets():
            print(f"\nResetting validation cycler at epoch {epoch}")
            validation_cycler.reset()
            validation_cycler.next_subset()
        
        # If either cycler was reset, create new dataloaders
        if (train_cycler and not train_cycler.has_more_subsets()) or \
           (validation_cycler and not validation_cycler.has_more_subsets()):
            train_loader, val_loader = create_subset_dataloaders(
                train_df=train_df,
                val_df=val_df,
                train_tickers=train_cycler.get_current_subset() if train_cycler else None,
                val_tickers=validation_cycler.get_current_subset() if validation_cycler else None,
                config=config,
                debug=debug
            )
            
            # Print current subset information
            if train_cycler:
                print(f"\nTrain subset {train_cycler.current_subset_idx + 1}/{len(train_cycler.subsets)}")
                print(f"Train tickers: {', '.join(train_cycler.get_current_subset())}")
            if validation_cycler:
                print(f"Validation subset {validation_cycler.current_subset_idx + 1}/{len(validation_cycler.subsets)}")
                print(f"Validation tickers: {', '.join(validation_cycler.get_current_subset())}")
        
        for batch_x, batch_y in train_loader:
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
                             train_params['max_checkpoints'], current_cycle=train_cycle)
            current_subset_checkpoints = []
            current_subset_start_epoch = epoch + 1
            
            # Increment cycle counter when we've gone through all subsets
            train_cycle += 1
        
        # Check early stopping
        if early_stop(current_val_loss):
            print(f"Early stopping triggered after epoch {epoch}")
            should_continue = False
            
            if enable_train_cycling:
                if train_cycler.has_more_subsets():
                    # Find best checkpoint from current subset with at least 2 epochs of training
                    valid_checkpoints = [(loss, ep, name) for loss, ep, name, epochs in current_subset_checkpoints if epochs >= 2]
                    
                    if valid_checkpoints:
                        # Select best checkpoint using new strategy
                        best_loss, best_epoch, best_name = select_best_checkpoint(
                            valid_checkpoints,
                            rewind_quantile_divisions,
                            rewind_min_extra_epochs
                        )
                        print(f"Loading checkpoint for next subset: Epoch {best_epoch}, Loss {best_loss:.7f}")
                        
                        checkpoint = torch.load(os.path.join(checkpoint_dir, best_name))
                        model.load_state_dict(checkpoint['model_state_dict'])
                        
                        # Important: Move model back to device after loading checkpoint
                        model = model.to(device)
                        
                        # Reinitialize optimizer with loaded model parameters
                        optimizer = torch.optim.AdamW(
                            model.parameters(),
                            lr=config['train_params']['scheduler']['base_lr'],
                            betas=(0.9, 0.95),
                            weight_decay=weight_decay
                        )
                        
                        # Reinitialize scheduler with new optimizer
                        scheduler = get_scheduler(optimizer, config, len(train_loader))
                        
                        if debug:
                            print("Model successfully loaded and moved to device")
                            print(f"Model device: {next(model.parameters()).device}")
                    
                    # Reset checkpoint tracking for new subset
                    current_subset_checkpoints = []
                    current_subset_start_epoch = epoch + 1
                    
                    # Move to next training subset and update validation subset
                    print(f"\nSwitching to next training subset after epoch {epoch}")
                    train_cycler.next_subset()
                    
                    # Also move to next validation subset if cycling is enabled
                    if enable_validation_cycling:
                        if not validation_cycler.has_more_subsets():
                            print(f"\nResetting validation cycler to beginning as training continues")
                            validation_cycler.reset_and_randomize()
                            val_cycle_completed = True
                        validation_cycler.next_subset()
                    
                    # Print current subset information
                    current_train_subset = train_cycler.get_current_subset()
                    print(f"\nSubset {train_cycler.current_subset_idx + 1}/{len(train_cycler.subsets)}")
                    print(f"Train tickers: {', '.join(current_train_subset)}")
                    if enable_validation_cycling:
                        current_val_subset = validation_cycler.get_current_subset()
                        print(f"Validation tickers: {', '.join(current_val_subset)}")
                    
                    # Create new dataloaders with updated subsets
                    train_loader, val_loader = create_subset_dataloaders(
                        train_df=train_df,
                        val_df=val_df,
                        train_tickers=train_cycler.get_current_subset(),
                        val_tickers=validation_cycler.get_current_subset() if enable_validation_cycling else None,
                        config=config,
                        debug=debug
                    )
                    
                    # Reset early stopping for new subset
                    early_stop.start_new_cycle()
                    should_continue = True
                    
                    if debug:
                        status = early_stop.get_improvement_status()
                        print(f"Starting new cycle. Best loss: {status['current_best_loss']:.7f}, Global best: {status['global_best_loss']:.7f}")
                        
                        # Add debug prints for model state
                        print("\nModel state after loading checkpoint:")
                        print(f"Model device: {next(model.parameters()).device}")
                        print(f"Optimizer state: {len(optimizer.state_dict()['state'])} parameter groups")
                        print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
                
                elif enable_validation_cycling and not val_cycle_completed:
                    # Train cycler is exhausted but validation hasn't completed a full cycle
                    # Reset train cycler and continue
                    print(f"\nResetting train cycler to continue validation cycling")
                    train_cycler.reset_and_randomize()
                    train_cycle_completed = True
                    train_cycler.next_subset()
                    
                    if not validation_cycler.has_more_subsets():
                        validation_cycler.reset_and_randomize()
                        val_cycle_completed = True
                    validation_cycler.next_subset()
                    
                    # Print current subset information
                    current_train_subset = train_cycler.get_current_subset()
                    print(f"\nSubset {train_cycler.current_subset_idx + 1}/{len(train_cycler.subsets)}")
                    print(f"Train tickers: {', '.join(current_train_subset)}")
                    current_val_subset = validation_cycler.get_current_subset()
                    print(f"Validation tickers: {', '.join(current_val_subset)}")
                    
                    # Create new dataloaders with updated subsets
                    train_loader, val_loader = create_subset_dataloaders(
                        train_df=train_df,
                        val_df=val_df,
                        train_tickers=train_cycler.get_current_subset(),
                        val_tickers=validation_cycler.get_current_subset(),
                        config=config,
                        debug=debug
                    )
                    
                    # Reset early stopping for new subset
                    early_stop.start_new_cycle()
                    should_continue = True
                    
                    if debug:
                        status = early_stop.get_improvement_status()
                        print(f"Starting new cycle. Best loss: {status['current_best_loss']:.7f}, Global best: {status['global_best_loss']:.7f}")
                else:
                    train_cycle_completed = True
            
            elif enable_validation_cycling and validation_cycler.has_more_subsets():
                # Only cycle validation if train cycling is not enabled
                print(f"\nSwitching to next validation subset after epoch {epoch}")
                validation_cycler.next_subset()
                
                # Print current subset information
                if train_cycler:
                    current_train_subset = train_cycler.get_current_subset()
                    print(f"Train tickers: {', '.join(current_train_subset)}")
                current_val_subset = validation_cycler.get_current_subset()
                print(f"Validation tickers: {', '.join(current_val_subset)}")
                
                # Start new cycle and check if we should stop globally
                if early_stop.start_new_cycle():
                    print(f"\nStopping training - No improvement for {early_stop.max_cycles_without_improvement} cycles")
                    print(f"Global best loss: {early_stop.global_best_loss:.7f}")
                    should_continue = False
                else:
                    val_loader = create_subset_dataloaders(
                        train_df=train_df,
                        val_df=val_df,
                        train_tickers=train_cycler.get_current_subset() if train_cycler else None,
                        val_tickers=validation_cycler.get_current_subset(),
                        config=config,
                        debug=debug
                    )[1]  # Only need val_loader
                    
                    should_continue = True
                    
                    if debug:
                        status = early_stop.get_improvement_status()
                        print(f"Starting new cycle. Best loss: {status['current_best_loss']:.7f}, Global best: {status['global_best_loss']:.7f}")
        
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
            final_checkpoint_name = f"{config['train_params']['prefix']}_final_id_{config['train_params']['run_id']}_arc_{config['train_params']['architecture_version']}_e{epoch}_valloss_{current_val_loss:.7f}.pt"
            torch.save(checkpoint, os.path.join(checkpoint_dir, final_checkpoint_name))
            print(f"Saved final chkpt: {final_checkpoint_name}")
            break
        
        # After validation, check if we should move to next subset
        if train_cycler and train_cycler.has_more_subsets():
            print(f"\nMoving to next training subset after epoch {epoch}")
            train_cycler.next_subset()
            
            if validation_cycler and validation_cycler.has_more_subsets():
                validation_cycler.next_subset()
            
            # Create new dataloaders with updated subsets
            train_loader, val_loader = create_subset_dataloaders(
                train_df=train_df,
                val_df=val_df,
                train_tickers=train_cycler.get_current_subset(),
                val_tickers=validation_cycler.get_current_subset() if validation_cycler else None,
                config=config,
                debug=debug
            )
            
            # Print current subset information
            print(f"\nTrain subset {train_cycler.current_subset_idx + 1}/{len(train_cycler.subsets)}")
            print(f"Train tickers: {', '.join(train_cycler.get_current_subset())}")
            if validation_cycler:
                print(f"Validation subset {validation_cycler.current_subset_idx + 1}/{len(validation_cycler.subsets)}")
                print(f"Validation tickers: {', '.join(validation_cycler.get_current_subset())}")
        
        # Print progress every print_every epochs
        if (epoch + 1) % train_params['print_every'] == 0:
            current_step = epoch * len(train_loader)  # Calculate current step
            print(f"Epoch: {epoch+1}/{n_epochs} (Step: {current_step})")
            print(f"Training loss: {train_losses[-1]:.7f}")
            print(f"Validation loss: {val_losses[-1]:.7f}")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
            
    # Save final checkpoint at end of training
    metadata = TrainingMetadata(
        epoch=n_epochs-1,
        val_loss=val_losses[-1],
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
    
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': metadata
    }
    
    final_checkpoint_name = f"{config['train_params']['prefix']}_final_id_{config['train_params']['run_id']}_arc_{config['train_params']['architecture_version']}_e{n_epochs-1}_valloss_{val_losses[-1]:.7f}.pt"
    torch.save(final_checkpoint, os.path.join(checkpoint_dir, final_checkpoint_name))
    print(f"Saved final checkpoint: {final_checkpoint_name}")
    
    return train_losses, val_losses
