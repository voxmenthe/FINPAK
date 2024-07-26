# flake8: noqa W293, E501
import numpy as np


# Function to calculate the distance from N-day high
def distance_from_high(data, n):
    # Calculate N-day highs
    n_day_highs = np.max(rolling_window(data, n), axis=1)
    
    # Current prices (we only consider the rows that have a complete N-day window)
    current_prices = data[n-1:]
    
    # Calculate the percentage difference from N-day high
    distancefh = ((current_prices - n_day_highs) / n_day_highs) * 100
    
    return distancefh


# Function to apply a rolling window operation
def rolling_window(a, window):
    """Create rolling window where time/period is first dimension"""
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_window_last(arr, window):
    """Create a rolling window where time/period is the last dimension"""
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def n_day_high_in_last_k_days(data: np.array, n: int, k: int):
    assert k >= n, "k must be greater than or equal to n"
    assert len(data.shape) == 2, "Input data must be a 2D numpy array"

    # Calculate the N-day high for each position
    n_day_highs = np.max(rolling_window(data, n), axis=1)

    # Get the last k days of n_day_highs
    last_k_days_highs = n_day_highs[-k:]

    # Calculate the maximum of the last k days for each ticker
    result = np.max(last_k_days_highs, axis=0)

    return result
