class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min'):
        """
        Early stopping to stop the training when the loss does not improve after
        certain epochs.
        
        Args:
            patience (int): Number of epochs to wait before stopping after last improvement
            min_delta (float): Minimum change in monitored value to qualify as an improvement
            mode (str): 'min' for loss, 'max' for metrics like accuracy
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
    
    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            return False
        
        if self.mode == 'min':
            delta = current_loss - self.best_loss
        else:
            delta = self.best_loss - current_loss
            
        if delta < self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        # Update recent losses
        self.recent_losses.append(current_loss)
        if len(self.recent_losses) > self.patience:
            self.recent_losses.pop(0)

        # Update best losses
        if len(self.best_losses) < self.patience:
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