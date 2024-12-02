import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass
import time
from ticker_cycler import TickerCycler
from functools import partial

@dataclass
class TrainingMetadata:
    epoch: int
    val_loss: float
    train_loss: float
    train_cycle: int
    val_cycle: int
    config: Dict[str, Any]
    training_history: Dict[str, list]
    timestamp: str
    trained_subsets: list

def load_checkpoint(checkpoint_path: Path, model: nn.Module, optimizer: optim.Optimizer) -> TrainingMetadata:
    """Load model checkpoint and return metadata"""
    checkpoint = mx.load(str(checkpoint_path))
    model.update(checkpoint['model_state'])
    optimizer.state.update(checkpoint['optimizer_state'])
    return checkpoint['metadata']

def train(
    model: nn.Module,
    train_loader: Callable,
    val_loader: Callable,
    optimizer: optim.Optimizer,
    loss_fn: Callable,
    n_epochs: int,
    save_dir: Path,
    save_freq: int = 1,
    early_stopping_patience: int = 10,
    min_delta: float = 1e-6,
    print_every: int = 10,
    max_checkpoints: int = 5,
    weight_decay: float = 0.0,
    scheduler: Optional[Dict] = None,
    train_cycler: Optional[TickerCycler] = None,
    val_cycler: Optional[TickerCycler] = None,
    current_cycle: int = 0,
    config: Optional[Dict] = None,
    trained_subsets: Optional[list] = None
) -> Dict[str, list]:
    """Train the model using MLX with subset cycling support"""
    
    save_dir.mkdir(parents=True, exist_ok=True)
    if trained_subsets is None:
        trained_subsets = []
    
    # Initialize training history
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Define loss computation
    def compute_loss(model, X, y):
        pred = model(X)
        return loss_fn(pred, y)
    
    # The state that will be captured as input and output
    state = [model.state, optimizer.state]
    
    # Define step function with state capture
    @partial(mx.compile, inputs=state, outputs=state)
    def step(X, y):
        loss, grads = nn.value_and_grad(model, compute_loss)(model, X, y)
        optimizer.update(model, grads)
        return loss
    
    # Define evaluation function
    @mx.compile
    def evaluate(X, y):
        pred = model(X)
        return loss_fn(pred, y)
    
    print("\nStarting training...")
    start_time = time.time()
    
    while True:  # Continue until all subsets are processed
        for epoch in range(n_epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_x, batch_y in train_loader():
                # Ensure inputs are MLX arrays
                if not isinstance(batch_x, mx.array):
                    batch_x = mx.array(batch_x)
                if not isinstance(batch_y, mx.array):
                    batch_y = mx.array(batch_y)
                
                # Update parameters and get loss
                loss = step(batch_x, batch_y)
                mx.eval(state)  # Evaluate the model and optimizer state
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches if n_batches > 0 else float('inf')
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            n_val_batches = 0
            
            for batch_x, batch_y in val_loader():
                # Ensure inputs are MLX arrays
                if not isinstance(batch_x, mx.array):
                    batch_x = mx.array(batch_x)
                if not isinstance(batch_y, mx.array):
                    batch_y = mx.array(batch_y)
                
                batch_loss = evaluate(batch_x, batch_y)
                val_loss += batch_loss.item()
                n_val_batches += 1
            
            val_loss = val_loss / n_val_batches if n_val_batches > 0 else float('inf')
            history['val_loss'].append(val_loss)
            
            # Print progress
            if (epoch + 1) % print_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{n_epochs} ({elapsed:.2f}s)")
                print(f"Train loss: {avg_train_loss:.6f}")
                print(f"Val loss: {val_loss:.6f}")
                if train_cycler:
                    print(f"Current train subset: {train_cycler.get_current_subset()}")
                if val_cycler:
                    print(f"Current val subset: {val_cycler.get_current_subset()}")
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save checkpoint
                if (epoch + 1) % save_freq == 0:
                    checkpoint_path = save_dir / f"checkpoint_tc{current_cycle}_e{epoch+1}_val{val_loss:.6f}.npz"
                    metadata = TrainingMetadata(
                        epoch=epoch,
                        val_loss=val_loss,
                        train_loss=avg_train_loss,
                        train_cycle=train_cycler.current_subset_idx if train_cycler else 0,
                        val_cycle=val_cycler.current_subset_idx if val_cycler else 0,
                        config=config,
                        training_history=history,
                        timestamp=time.strftime("%Y%m%d-%H%M%S"),
                        trained_subsets=trained_subsets + [train_cycler.get_current_subset()] if train_cycler else []
                    )
                    
                    mx.savez(
                        str(checkpoint_path),
                        model_state=model.state,
                        optimizer_state=optimizer.state,
                        metadata=metadata.__dict__
                    )
                    print(f"\nSaved checkpoint: {checkpoint_path.name}")
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after epoch {epoch+1}")
                
                # Check if we should move to next subset
                train_completed = True if not train_cycler else train_cycler.next_subset()
                val_completed = True if not val_cycler else val_cycler.next_subset()
                
                if train_completed and val_completed:
                    print("\nCompleted all subsets!")
                    return history
                
                if train_cycler or val_cycler:
                    print("\nMoving to next subset...")
                    # Reset early stopping for new subset
                    patience_counter = 0
                    best_val_loss = float('inf')
                    break
        
        # Check if we've completed all subsets
        train_completed = True if not train_cycler else train_cycler.has_more_subsets()
        val_completed = True if not val_cycler else val_cycler.has_more_subsets()
        
        if not train_completed or not val_completed:
            # Reset for next subset
            patience_counter = 0
            best_val_loss = float('inf')
            continue
        else:
            break
    
    return history
