import numpy as np
import pandas as pd
from financial_analysis.analysis.screeners.stock_criteria import count_stocks_above_moving_average, count_stocks_below_moving_average, count_stocks_hit_n_day_high


def timeseries_stocks_above_moving_average(data, window):
    """
    Generate a time series of the number of stocks above their moving average.

    Parameters:
    data (pd.DataFrame): DataFrame where each column is a stock's price series.
    window (int): The lookback period for the moving average.

    Returns:
    pd.Series: Time series of the number of stocks above their moving average.
    """
    moving_averages = data.rolling(window=window).mean()
    above_ma = (data > moving_averages).sum(axis=1)
    return above_ma

def timeseries_stocks_below_moving_average(data, window):
    """
    Generate a time series of the number of stocks below their moving average.

    Parameters:
    data (pd.DataFrame): DataFrame where each column is a stock's price series.
    window (int): The lookback period for the moving average.

    Returns:
    pd.Series: Time series of the number of stocks below their moving average.
    """
    moving_averages = data.rolling(window=window).mean()
    below_ma = (data < moving_averages).sum(axis=1)
    return below_ma

def timeseries_stocks_hit_n_day_high(data, n, k):
    """
    Generate a time series of the number of stocks that hit an N-day high within the past K days.

    Parameters:
    data (pd.DataFrame): DataFrame where each column is a stock's price series.
    n (int): The lookback period for the N-day high.
    k (int): The number of days to look back for the N-day high.

    Returns:
    pd.Series: Time series of the number of stocks that hit an N-day high within the past K days.
    """
    n_day_highs = data.rolling(window=n).max()
    hit_n_day_high = pd.Series(index=data.index, dtype=int)
    for i in range(len(data)):
        if i >= k:
            recent_highs = n_day_highs.iloc[i-k:i]
            hit_n_day_high.iloc[i] = (recent_highs.max() == data.iloc[i]).sum()
        else:
            hit_n_day_high.iloc[i] = 0
    return hit_n_day_high