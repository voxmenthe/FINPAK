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
                
        return self.early_stop 