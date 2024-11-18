from typing import List, Optional
import numpy as np
import random


class TickerCycler:
    def __init__(self,
            tickers: List[str],
            subset_size: int,
            overlap_size: Optional[int] = None,
            reverse_tickers: bool = False,
            use_anchor: bool = False,
            random_seed: Optional[int] = None):
        """
        Initialize the validation set cycler.
        
        Args:
            tickers: List of all validation tickers
            subset_size: Size of each validation subset (including first ticker)
            overlap_size: Number of tickers to overlap between consecutive subsets.
                         If None, will use subset_size//3 as default
            reverse_tickers: If True, use the last ticker as the first ticker
            use_anchor: If True, always include first_ticker in each subset
            random_seed: Optional seed for random number generation
        """
        if len(tickers) < 2:
            raise ValueError("Need at least 2 tickers for cycling")
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            
        # Set up first ticker and remaining tickers
        if use_anchor:
            if reverse_tickers:
                self.first_ticker = tickers[-1]
                self.all_tickers = list(tickers[:-1])  # Convert to list for random.sample
            else:
                self.first_ticker = tickers[0]
                self.all_tickers = list(tickers[1:])
        else:
            self.all_tickers = list(tickers)
        
        self.use_anchor = use_anchor
        
        # Adjust subset size to account for first ticker
        self.subset_size = min(subset_size, len(tickers))
        self.adjusted_subset_size = self.subset_size - 1 if use_anchor else self.subset_size
        
        # Set overlap size (must be less than adjusted subset size)
        if overlap_size is None:
            self.overlap_size = max(1, self.adjusted_subset_size // 3)
        else:
            self.overlap_size = min(overlap_size, self.adjusted_subset_size - 1)
        
        # Generate all possible subsets
        self.subsets = self._generate_subsets()
        self.current_subset_idx = 0
        
    def _generate_subsets(self) -> List[List[str]]:
        """Generate subsets with random overlap between consecutive subsets."""
        n_tickers = len(self.all_tickers)
        subsets = []
        
        # Handle case where adjusted subset size >= number of remaining tickers
        if self.adjusted_subset_size >= n_tickers:
            subset = self.all_tickers.copy()
            if self.use_anchor:
                subset = [self.first_ticker] + subset
            return [subset]
        
        # First subset is random sample
        current_subset = random.sample(self.all_tickers, self.adjusted_subset_size)
        if self.use_anchor:
            current_subset = [self.first_ticker] + current_subset
        subsets.append(current_subset)
        
        # Generate remaining subsets
        remaining_tickers = set(self.all_tickers)
        while len(subsets) < (n_tickers // (self.adjusted_subset_size - self.overlap_size)):
            # Randomly select tickers to keep from previous subset
            prev_subset = current_subset[1:] if self.use_anchor else current_subset
            keep_tickers = random.sample(prev_subset, self.overlap_size)
            
            # Select remaining tickers randomly from those not in previous subset
            available_tickers = list(remaining_tickers - set(prev_subset))
            if len(available_tickers) < (self.adjusted_subset_size - self.overlap_size):
                # If we don't have enough new tickers, reset the available pool
                available_tickers = list(remaining_tickers - set(keep_tickers))
            
            new_tickers = random.sample(
                available_tickers,
                self.adjusted_subset_size - self.overlap_size
            )
            
            # Combine kept and new tickers
            current_subset = keep_tickers + new_tickers
            if self.use_anchor:
                current_subset = [self.first_ticker] + current_subset
                
            subsets.append(current_subset)
            
        return subsets
    
    def get_current_subset(self) -> List[str]:
        """Get the current validation subset."""
        return self.subsets[self.current_subset_idx]
    
    def next_subset(self) -> List[str]:
        """Move to the next subset and return it."""
        self.current_subset_idx = (self.current_subset_idx + 1) % len(self.subsets)
        return self.get_current_subset()
    
    def reset(self):
        """Reset to the first subset."""
        self.current_subset_idx = 0
        
    def has_more_subsets(self) -> bool:
        """Check if there are more unused subsets in the current cycle."""
        return self.current_subset_idx < len(self.subsets) - 1
