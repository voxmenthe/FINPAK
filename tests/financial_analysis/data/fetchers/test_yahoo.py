import unittest
import pandas as pd

from financial_analysis.data.fetchers.yahoo import (
    download_historical_data,
    download_multiple_tickers,
    save_data_to_file,
    update_data_file
)


class TestYahooFetchers(unittest.TestCase):

    def test_download_historical_data(self):
        data = download_historical_data('AAPL', '2020-01-01', '2020-01-10')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

    def test_download_multiple_tickers(self):
        data = download_multiple_tickers(['AAPL', 'MSFT'], '2020-01-01', '2020-01-10')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertIn('AAPL', data.columns.get_level_values(1))
        self.assertIn('MSFT', data.columns.get_level_values(1))

    def test_save_data_to_file_csv(self):
        data = download_historical_data('AAPL', '2020-01-01', '2020-01-10')
        save_data_to_file(data, 'test_data.csv', 'csv')
        loaded_data = pd.read_csv('test_data.csv', index_col=0, parse_dates=True)
        pd.testing.assert_frame_equal(data, loaded_data)

    def test_save_data_to_file_parquet(self):
        data = download_historical_data('AAPL', '2020-01-01', '2020-01-10')
        save_data_to_file(data, 'test_data.parquet', 'parquet')
        loaded_data = pd.read_parquet('test_data.parquet')
        pd.testing.assert_frame_equal(data, loaded_data)

    def test_update_data_file_csv(self):
        data = download_historical_data('AAPL', '2020-01-01', '2020-01-10')
        save_data_to_file(data, 'test_update_data.csv', 'csv')
        update_data_file('AAPL', 'test_update_data.csv', 'csv')
        updated_data = pd.read_csv('test_update_data.csv', index_col=0, parse_dates=True)
        self.assertFalse(updated_data.empty)

    def test_update_data_file_parquet(self):
        data = download_historical_data('AAPL', '2020-01-01', '2020-01-10')
        save_data_to_file(data, 'test_update_data.parquet', 'parquet')
        update_data_file('AAPL', 'test_update_data.parquet', 'parquet')
        updated_data = pd.read_parquet('test_update_data.parquet')
        self.assertFalse(updated_data.empty)


if __name__ == '__main__':
    unittest.main()
