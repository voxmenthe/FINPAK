import unittest
import pandas as pd
import numpy as np
from analysis.indicators.technical.trend import (
    donchian_channels,
    moving_average,
    linear_regression_channel,
    linear_regression_curve,
    exponential_moving_average,
    bollinger_bands,
    parabolic_sar,
    average_directional_index,
    ichimoku_cloud
)

class TestTrendIndicators(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range('2020-01-01', periods=100)
        self.data = pd.DataFrame({
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100
        }, index=dates)

    def test_donchian_channels(self):
        result = donchian_channels(self.data['close'], window=20)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('upper', result.columns)
        self.assertIn('middle', result.columns)
        self.assertIn('lower', result.columns)

    def test_moving_average(self):
        result = moving_average(self.data['close'], window=20)
        self.assertIsInstance(result, pd.Series)

    def test_linear_regression_channel(self):
        result = linear_regression_channel(self.data['close'], window=20)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('upper', result.columns)
        self.assertIn('middle', result.columns)
        self.assertIn('lower', result.columns)

    def test_linear_regression_curve(self):
        result = linear_regression_curve(self.data['close'], window=20)
        self.assertIsInstance(result, pd.Series)

    def test_exponential_moving_average(self):
        result = exponential_moving_average(self.data['close'], window=20)
        self.assertIsInstance(result, pd.Series)

    def test_bollinger_bands(self):
        result = bollinger_bands(self.data['close'], window=20)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('upper', result.columns)
        self.assertIn('middle', result.columns)
        self.assertIn('lower', result.columns)

    def test_parabolic_sar(self):
        result = parabolic_sar(self.data)
        self.assertIsInstance(result, pd.Series)

    def test_average_directional_index(self):
        result = average_directional_index(self.data, window=20)
        self.assertIsInstance(result, pd.Series)

    def test_ichimoku_cloud(self):
        result = ichimoku_cloud(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('tenkan_sen', result.columns)
        self.assertIn('kijun_sen', result.columns)
        self.assertIn('senkou_span_a', result.columns)
        self.assertIn('senkou_span_b', result.columns)
        self.assertIn('chikou_span', result.columns)

if __name__ == '__main__':
    unittest.main()
