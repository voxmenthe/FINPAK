import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim
from typing import Dict, Tuple, Callable
import time
from pathlib import Path
from functools import partial

def create_train_step(
    model: mlx_nn.Module,
    optimizer: mlx_optim.Optimizer,
    loss_fn: Callable
) -> Tuple[Callable, Callable]:
    """
    Creates training and evaluation functions using MLX's function transforms.
    """
    def loss_fn_wrapper(model, X, y):
        """Base loss computation"""
        return loss_fn(model(X), y)
    
    # Create training and evaluation functions with MLX transforms
    loss_and_grad_fn = mlx_nn.value_and_grad(model, loss_fn_wrapper)
    
    # Capture all state that needs to be updated
    state = [model.state, optimizer.state]
    
    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(X, y):
        """Performs a single training step."""
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        return loss
    
    @partial(mx.compile, inputs=[model.state])
    def eval_step(X, y):
        """Performs a single evaluation step."""
        return loss_fn(model(X), y)
    
    return train_step, eval_step

def train(
    model: mlx_nn.Module,
    train_loader: Callable,
    val_loader: Callable,
    optimizer: mlx_optim.Optimizer,
    loss_fn: Callable,
    n_epochs: int,
    save_dir: str,
    save_freq: int = 1,
    early_stopping_patience: int = 10,
    scheduler = None
) -> Dict:
    """Training loop for MLX model."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training and evaluation functions
    train_step, eval_step = create_train_step(model, optimizer, loss_fn)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch_times': []
    }
    
    # Initialize model parameters
    mx.eval(model.parameters())
    
    # Capture state for updates
    state = [model.state, optimizer.state]
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        # Training loop
        train_losses = []
        for batch_x, batch_y in train_loader():
            # Convert numpy arrays to MLX arrays if needed
            if not isinstance(batch_x, mx.array):
                batch_x = mx.array(batch_x)
            if not isinstance(batch_y, mx.array):
                batch_y = mx.array(batch_y)
                
            # Compute loss and update model
            loss = train_step(batch_x, batch_y)
            mx.eval(state)  # Evaluate all state at once
            train_losses.append(loss.item())
        
        # Validation loop
        val_losses = []
        for batch_x, batch_y in val_loader():
            if not isinstance(batch_x, mx.array):
                batch_x = mx.array(batch_x)
            if not isinstance(batch_y, mx.array):
                batch_y = mx.array(batch_y)
                
            val_loss = eval_step(batch_x, batch_y)
            mx.eval(val_loss)
            val_losses.append(val_loss.item())
        
        # Calculate epoch metrics
        epoch_train_loss = sum(train_losses) / len(train_losses)
        epoch_val_loss = sum(val_losses) / len(val_losses)
        epoch_time = time.time() - epoch_start
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['epoch_times'].append(epoch_time)
        
        # Print progress
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        
        # Save checkpoint if improved
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            
            if (epoch + 1) % save_freq == 0:
                mx.eval(model.parameters())
                model.save_weights(str(save_dir / f"model_epoch_{epoch+1}.npz"))
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Update learning rate if scheduler provided
        if scheduler is not None:
            scheduler.step()
            mx.eval(scheduler.state)
    
    return history
