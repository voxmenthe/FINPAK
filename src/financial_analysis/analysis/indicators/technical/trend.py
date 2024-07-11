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

def exponential_moving_average(data, window, offset=0):
    """
    Calculate Exponential Moving Average (EMA).

    Parameters:
    data (pd.Series): The input price series.
    window (int): The lookback period for the EMA.
    offset (int): The number of periods to offset the result.

    Returns:
    pd.Series: The EMA.
    """
    return data.ewm(span=window, adjust=False).mean().shift(offset)

def bollinger_bands(data, window, num_std_dev=2, offset=0):
    """
    Calculate Bollinger Bands.

    Parameters:
    data (pd.Series): The input price series.
    window (int): The lookback period for the moving average.
    num_std_dev (int): The number of standard deviations for the bands.
    offset (int): The number of periods to offset the result.

    Returns:
    pd.DataFrame: The Bollinger Bands (upper, lower, and middle).
    """
    middle = data.rolling(window=window).mean().shift(offset)
    std_dev = data.rolling(window=window).std().shift(offset)
    upper = middle + (num_std_dev * std_dev)
    lower = middle - (num_std_dev * std_dev)
    return pd.DataFrame({'upper': upper, 'middle': middle, 'lower': lower})

def parabolic_sar(data, step=0.02, max_step=0.2, offset=0):
    """
    Calculate Parabolic SAR.

    Parameters:
    data (pd.DataFrame): The input price data with 'high' and 'low' columns.
    step (float): The step increment.
    max_step (float): The maximum step.
    offset (int): The number of periods to offset the result.

    Returns:
    pd.Series: The Parabolic SAR.
    """
    high = data['high']
    low = data['low']
    sar = np.zeros(len(data))
    trend = 1
    af = step
    ep = high[0]
    sar[0] = low[0]

    for i in range(1, len(data)):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        if trend == 1:
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
            if low[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = low[i]
                af = step
        else:
            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)
            if high[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = step

    return pd.Series(sar, index=data.index).shift(offset)

def average_directional_index(data, window, offset=0):
    """
    Calculate Average Directional Index (ADX).

    Parameters:
    data (pd.DataFrame): The input price data with 'high', 'low', and 'close' columns.
    window (int): The lookback period for the ADX.
    offset (int): The number of periods to offset the result.

    Returns:
    pd.Series: The ADX.
    """
    high = data['high']
    low = data['low']
    close = data['close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (abs(minus_dm).rolling(window=window).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=window).mean()
    return adx.shift(offset)

def ichimoku_cloud(data, offset=0):
    """
    Calculate Ichimoku Cloud.

    Parameters:
    data (pd.DataFrame): The input price data with 'high' and 'low' columns.
    offset (int): The number of periods to offset the result.

    Returns:
    pd.DataFrame: The Ichimoku Cloud (tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span).
    """
    high = data['high']
    low = data['low']
    tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
    chikou_span = data['close'].shift(-26)
    return pd.DataFrame({
        'tenkan_sen': tenkan_sen.shift(offset),
        'kijun_sen': kijun_sen.shift(offset),
        'senkou_span_a': senkou_span_a.shift(offset),
        'senkou_span_b': senkou_span_b.shift(offset),
        'chikou_span': chikou_span.shift(offset)
    })
