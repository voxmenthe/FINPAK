import numpy as np
import pandas as pd
from scipy.stats import linregress

def donchian_channels(data, window, offset=0):
    """
    Calculate Donchian Channels.

    Parameters:
    data (pd.Series): The input price series.
    window (int): The lookback period for the channels.
    offset (int): The number of periods to offset the result.

    Returns:
    pd.DataFrame: The Donchian Channels (upper, lower, and middle).
    """
    upper = data.rolling(window=window).max().shift(offset)
    lower = data.rolling(window=window).min().shift(offset)
    middle = (upper + lower) / 2
    return pd.DataFrame({'upper': upper, 'middle': middle, 'lower': lower})

def moving_average(data, window, offset=0):
    """
    Calculate Moving Average.

    Parameters:
    data (pd.Series): The input price series.
    window (int): The lookback period for the moving average.
    offset (int): The number of periods to offset the result.

    Returns:
    pd.Series: The moving average.
    """
    return data.rolling(window=window).mean().shift(offset)

def linear_regression_channel(data, window, offset=0):
    """
    Calculate Linear Regression Channel.

    Parameters:
    data (pd.Series): The input price series.
    window (int): The lookback period for the linear regression.
    offset (int): The number of periods to offset the result.

    Returns:
    pd.DataFrame: The Linear Regression Channel (upper, lower, and middle).
    """
    x = np.arange(window)
    slopes = []
    intercepts = []
    for i in range(len(data) - window + 1):
        y = data[i:i+window]
        slope, intercept, _, _, _ = linregress(x, y)
        slopes.append(slope)
        intercepts.append(intercept)
    slopes = np.array(slopes)
    intercepts = np.array(intercepts)
    middle = intercepts + slopes * (window - 1)
    upper = middle + data.rolling(window=window).std().shift(offset)
    lower = middle - data.rolling(window=window).std().shift(offset)
    return pd.DataFrame({'upper': upper, 'middle': middle, 'lower': lower})

def linear_regression_curve(data, window, offset=0):
    """
    Calculate Linear Regression Curve.

    Parameters:
    data (pd.Series): The input price series.
    window (int): The lookback period for the linear regression.
    offset (int): The number of periods to offset the result.

    Returns:
    pd.Series: The linear regression curve.
    """
    x = np.arange(window)
    regression_values = []
    for i in range(len(data) - window + 1):
        y = data[i:i+window]
        slope, intercept, _, _, _ = linregress(x, y)
        regression_values.append(intercept + slope * (window - 1))
    regression_values = np.array(regression_values)
    return pd.Series(regression_values, index=data.index[window-1:]).shift(offset)
