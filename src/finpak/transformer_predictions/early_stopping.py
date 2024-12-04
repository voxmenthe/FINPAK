class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 max_checkpoints: int = 5, min_epochs: int = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.min_epochs = min_epochs
        self.max_checkpoints = max_checkpoints
        self.cycles_without_improvement = 0
        self.max_cycles_without_improvement = 3
        self.current_cycle_best = float('inf')
        self.global_best_loss = float('inf')
        
    def is_best(self, val_loss: float) -> bool:
        """
        Check if the current validation loss is the best so far.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            bool: True if current loss is better than previous best
        """
        if val_loss < self.best_loss - self.min_delta:
            return True
        return False
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            bool: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
        
    def start_new_cycle(self) -> None:
        """Reset counters for a new training cycle while preserving global best."""
        self.counter = 0
        self.best_loss = float('inf')
        if self.current_cycle_best > self.global_best_loss:
            self.cycles_without_improvement += 1
        else:
            self.cycles_without_improvement = 0
            self.global_best_loss = self.current_cycle_best
        self.current_cycle_best = float('inf')
