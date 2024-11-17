from typing import List, Optional
import numpy as np


class TickerCycler:
    def __init__(self,
            tickers: List[str],
            subset_size: int,
            overlap_size: Optional[int] = None,
            reverse_tickers: bool = False,
            use_anchor: bool = False):
        """
        Initialize the validation set cycler.
        
        Args:
            tickers: List of all validation tickers
            subset_size: Size of each validation subset (including first ticker)
            overlap_size: Number of tickers to overlap between consecutive subsets.
                         If None, will use subset_size//3 as default
            reverse_tickers: If True, use the last ticker as the first ticker
        """
        if len(tickers) < 2:
            raise ValueError("Need at least 2 tickers for cycling")
            
        # Set up first ticker and remaining tickers
        if use_anchor:
            if reverse_tickers:
                self.first_ticker = tickers[-1]
                self.all_tickers = tickers[:-1]
            else:
                self.first_ticker = tickers[0]
                self.all_tickers = tickers[1:]
        else:
            self.all_tickers = tickers
        
        self.use_anchor = use_anchor
        
        # Adjust subset size to account for first ticker
        self.subset_size = min(subset_size, len(tickers))
        self.adjusted_subset_size = self.subset_size - 1  # Space for other tickers
        
        # Set overlap size (must be less than adjusted subset size)
        if overlap_size is None:
            self.overlap_size = max(1, self.adjusted_subset_size // 3)
        else:
            self.overlap_size = min(overlap_size, self.adjusted_subset_size - 1)
        
        # Calculate step size between subsets using adjusted size
        self.step_size = self.adjusted_subset_size - self.overlap_size
        
        # Generate all possible subsets
        self.subsets = self._generate_subsets()
        self.current_subset_idx = 0
        
    def _generate_subsets(self) -> List[List[str]]:
        """Generate all possible subsets with the specified overlap."""
        n_tickers = len(self.all_tickers)
        subsets = []
        
        # Handle case where adjusted subset size >= number of remaining tickers
        if self.adjusted_subset_size >= n_tickers:
            return [[self.first_ticker] + self.all_tickers]
            
        start_idx = 0
        while True:
            # Get current subset without first ticker
            subset = []
            for i in range(start_idx, start_idx + self.adjusted_subset_size):
                # Use modulo to wrap around to the beginning of the list
                subset.append(self.all_tickers[i % n_tickers])
            
            if self.use_anchor:
                # Add first ticker at the beginning
                subset = [self.first_ticker] + subset
            
            subsets.append(subset)
            
            # Move start index by step_size
            start_idx += self.step_size
            
            # If we've wrapped around completely, stop
            if start_idx % n_tickers < self.step_size:
                break
        
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
