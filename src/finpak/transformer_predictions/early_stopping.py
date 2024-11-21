class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min', max_checkpoints=3, min_epochs=0):
        """
        Early stopping to halt training when the loss doesn't improve after
        certain epochs, with additional features:
        
        1. Maintains a list of best models
        2. Enforces a minimum number of epochs before stopping
        3. Tracks recent losses for trend analysis
        4. Tracks both cycle-specific and global best losses
        
        Args:
            patience (int): Number of epochs to wait before stopping after last improvement
            min_delta (float): Minimum change in monitored value to qualify as an improvement
            mode (str): 'min' for loss, 'max' for metrics like accuracy
            max_checkpoints (int): Maximum number of best checkpoints to keep
            min_epochs (int): Minimum number of epochs before allowing early stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None  # Best loss for current cycle
        self.global_best_loss = None  # Best loss across all cycles
        self.early_stop = False
        self.min_delta *= 1 if mode == 'min' else -1
        self.best_losses = []  # Track the best losses
        self.recent_losses = []  # Track the recent losses
        self.max_checkpoints = max_checkpoints
        self.min_epochs = min_epochs
        self.current_epoch = 0
        self.epochs_since_improvement = 0
        self.last_improvement_epoch = 0
        self.cycles_without_improvement = 0  # Track cycles without improvement
        self.max_cycles_without_improvement = 2  # Stop after this many cycles without improvement

    def __call__(self, current_loss):
        self.current_epoch += 1
        
        # Don't allow early stopping before minimum epochs
        if self.current_epoch < self.min_epochs:
            return False

        # Initialize best losses if this is the first call
        if self.best_loss is None:
            self.best_loss = current_loss
            self.global_best_loss = current_loss if self.global_best_loss is None else self.global_best_loss
            self.last_improvement_epoch = self.current_epoch
            return False
        
        if self.mode == 'min':
            delta = current_loss - self.best_loss
            global_delta = current_loss - self.global_best_loss
        else:
            delta = self.best_loss - current_loss
            global_delta = self.global_best_loss - current_loss
            
        # Check for improvement in current cycle
        if delta < self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.epochs_since_improvement = 0
            self.last_improvement_epoch = self.current_epoch
            
            # Update global best if this is better
            if global_delta < self.min_delta:
                self.global_best_loss = current_loss
                self.cycles_without_improvement = 0
        else:
            self.counter += 1
            self.epochs_since_improvement += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        # Update recent losses
        self.recent_losses.append(current_loss)
        if len(self.recent_losses) > self.patience:
            self.recent_losses.pop(0)

        # Update best losses list
        if len(self.best_losses) < self.max_checkpoints:
            self.best_losses.append(current_loss)
            self.best_losses.sort(reverse=(self.mode == 'max'))
        else:
            if (self.mode == 'min' and current_loss < self.best_losses[-1]) or \
               (self.mode == 'max' and current_loss > self.best_losses[-1]):
                self.best_losses[-1] = current_loss
                self.best_losses.sort(reverse=(self.mode == 'max'))

        return self.early_stop

    def start_new_cycle(self):
        """Reset cycle-specific tracking while maintaining global tracking"""
        if self.best_loss is not None:  # Only count cycles where we had some training
            if self.mode == 'min':
                delta = self.best_loss - self.global_best_loss
            else:
                delta = self.global_best_loss - self.best_loss
                
            # If this cycle didn't improve on global best, increment counter
            if delta >= self.min_delta:
                self.cycles_without_improvement += 1
        
        # Reset cycle-specific tracking
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.recent_losses = []
        
        # If we haven't improved for too many cycles, signal to stop
        return self.cycles_without_improvement >= self.max_cycles_without_improvement

    def get_improvement_status(self):
        return {
            'epochs_since_improvement': self.epochs_since_improvement,
            'last_improvement_epoch': self.last_improvement_epoch,
            'early_stop': self.early_stop,
            'cycles_without_improvement': self.cycles_without_improvement,
            'current_best_loss': self.best_loss,
            'global_best_loss': self.global_best_loss
        }
