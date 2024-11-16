class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min', max_checkpoints=3, min_epochs=0):
        """
        Early stopping to halt training when the loss doesn't improve after
        certain epochs, with additional features:
        
        1. Maintains a list of best models
        2. Enforces a minimum number of epochs before stopping
        3. Tracks recent losses for trend analysis
        
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
        self.best_loss = None
        self.early_stop = False
        self.min_delta *= 1 if mode == 'min' else -1
        self.best_losses = []  # Track the best losses
        self.recent_losses = []  # Track the recent losses
        self.max_checkpoints = max_checkpoints  # Use max_checkpoints for best losses
        self.min_epochs = min_epochs
        self.current_epoch = 0
        self.epochs_since_improvement = 0  # Track epochs since last improvement
        self.last_improvement_epoch = 0  # Track when we last saw an improvement

    def __call__(self, current_loss):
        self.current_epoch += 1
        
        # Don't allow early stopping before minimum epochs
        if self.current_epoch < self.min_epochs:
            return False

        if self.best_loss is None:
            self.best_loss = current_loss
            self.last_improvement_epoch = self.current_epoch
            return False
        
        if self.mode == 'min':
            delta = current_loss - self.best_loss
        else:
            delta = self.best_loss - current_loss
            
        if delta < self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.epochs_since_improvement = 0
            self.last_improvement_epoch = self.current_epoch
        else:
            self.counter += 1
            self.epochs_since_improvement += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        # Update recent losses
        self.recent_losses.append(current_loss)
        if len(self.recent_losses) > self.patience:
            self.recent_losses.pop(0)

        # Update best losses
        if len(self.best_losses) < self.max_checkpoints:
            self.best_losses.append(current_loss)
            self.best_losses.sort(reverse=(self.mode == 'max'))
        else:
            if (self.mode == 'min' and current_loss < self.best_losses[-1]) or \
               (self.mode == 'max' and current_loss > self.best_losses[-1]):
                self.best_losses[-1] = current_loss
                self.best_losses.sort(reverse=(self.mode == 'max'))

        # Check if none of the recent losses are in the best losses
        if all(loss not in self.best_losses for loss in self.recent_losses):
            self.early_stop = True

        return self.early_stop

    def get_improvement_status(self):
        return {
            'epochs_since_improvement': self.epochs_since_improvement,
            'last_improvement_epoch': self.last_improvement_epoch,
            'early_stop': self.early_stop
        }
