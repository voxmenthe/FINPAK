import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from torch.utils.data import DataLoader
import os
from heapq import heappush, heappushpop
from early_stopping import EarlyStopping
from torch.amp import GradScaler, autocast
import torch.distributed as dist


def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD.
        backend: The chosen backend for the orthogonalization step.
        backend_steps: The number of iteration steps to use in the backend.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, 
                       backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            zeropower_backend = zeropower_backends[group['backend']]
            backend_steps = group['backend_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                g = p.grad
                state = self.state[p]
                
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                g = zeropower_backend(g, steps=backend_steps)
                g *= max(1, g.size(0)/g.size(1))**0.5
                
                p.data.add_(g, alpha=-lr)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    warmup_steps: int = 1000,
    max_checkpoints: int = 3,
    checkpoint_dir: str = "checkpoints",
    prefix: str = 'mpv',
    patience: int = 7,
    min_delta: float = 0.0,
    start_epoch: int = 0,
    use_mixed_precision: bool = False
) -> Tuple[List[float], List[float]]:
    """
    Train the model with learning rate warmup and Muon optimizer
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    early_stop = EarlyStopping(patience=patience, min_delta=min_delta, max_checkpoints=max_checkpoints)
    
    # Initialize heap for keeping track of best models
    best_models = []
    
    # Initialize scaler for mixed precision
    scaler = GradScaler() if use_mixed_precision else None

    model = model.to(device)
    
    # Split parameters into matrix and non-matrix parameters
    matrix_params = [p for p in model.parameters() if p.ndim == 2]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    
    # Use Muon for matrix parameters and Adam for scalar parameters
    optimizer1 = Muon(matrix_params, lr=learning_rate, momentum=0.95)
    optimizer2 = torch.optim.AdamW(
        scalar_params,
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + np.cos((step - warmup_steps) * np.pi / n_epochs))
        
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda)
    scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lr_lambda)
    
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
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            
            if use_mixed_precision:
                with autocast():
                    predictions = model(batch_x)
                    loss = F.mse_loss(predictions, batch_y)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer1)
                scaler.unscale_(optimizer2)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer1)
                scaler.step(optimizer2)
                scaler.update()
            else:
                # Standard training
                predictions = model(batch_x)
                loss = F.mse_loss(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer1.step()
                optimizer2.step()
            
            scheduler1.step()
            scheduler2.step()
            epoch_loss += loss.item()
            
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                if use_mixed_precision:
                    with autocast():
                        predictions = model(batch_x)
                        val_loss += F.mse_loss(predictions, batch_y).item()
                else:
                    predictions = model(batch_x)
                    val_loss += F.mse_loss(predictions, batch_y).item()
                
        current_val_loss = val_loss / len(val_loader)
        val_losses.append(current_val_loss)
        
        # Save checkpoint if it's among the best
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer1.state_dict(),
            'optimizer_state_dict': optimizer2.state_dict(),
            'val_loss': current_val_loss,
            'train_loss': train_losses[-1]
        }
        
        # Add scaler state only if using mixed precision
        if use_mixed_precision:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
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
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"Training loss: {train_losses[-1]:.7f}")
            print(f"Validation loss: {val_losses[-1]:.7f}")
            print(f"Learning rate: {scheduler1.get_last_lr()[0]:.2e}")
            
    # At the end of training, print information about best checkpoints
    print("\nBest checkpoints:")
    for neg_loss, epoch in sorted(best_models, reverse=True):
        print(f"Epoch {epoch}: validation loss = {-neg_loss:.7f}")
            
    return train_losses, val_losses
