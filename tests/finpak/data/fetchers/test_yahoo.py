import unittest
import os
import pandas as pd
from datetime import date
from unittest.mock import patch, MagicMock

from data.fetchers.yahoo import (
    download_historical_data,
    download_multiple_tickers,
    save_data_to_file,
    check_existing_data,
    update_data_file
)


class TestYahooFetchers(unittest.TestCase):

    @patch('data.fetchers.yahoo.yf.download')
    def test_download_historical_data(self, mock_download):
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', end='2020-01-10'),
            'Close': range(10)
        }).set_index('Date')
        mock_download.return_value = mock_data

        data = download_historical_data('AAPL', '2020-01-01', '2020-01-10')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)

    @patch('data.fetchers.yahoo.yf.download')
    def test_download_multiple_tickers(self, mock_download):
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', end='2020-01-10'),
            'AAPL': range(10),
            'MSFT': range(10, 20)
        }).set_index('Date')
        mock_download.return_value = mock_data

        data = download_multiple_tickers(['AAPL', 'MSFT'], '2020-01-01', '2020-01-10')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertIn('AAPL', data.columns)
        self.assertIn('MSFT', data.columns)

    def test_save_data_to_file_csv(self):
        data = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', end='2020-01-10'),
            'Close': range(10)
        }).set_index('Date')
        save_data_to_file(data, 'AAPL', 'csv')
        folder_path = os.path.join('data_store', 'AAPL')
        filename = os.path.join(folder_path, f"{date.today().strftime('%Y-%m-%d')}.csv")
        loaded_data = pd.read_csv(filename, index_col=0, parse_dates=True)
        pd.testing.assert_frame_equal(data, loaded_data)

    def test_save_data_to_file_parquet(self):
        data = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', end='2020-01-10'),
            'Close': range(10)
        }).set_index('Date')
        save_data_to_file(data, 'AAPL', 'parquet')
        folder_path = os.path.join('data_store', 'AAPL')
        filename = os.path.join(folder_path, f"{date.today().strftime('%Y-%m-%d')}.parquet")
        loaded_data = pd.read_parquet(filename)
        pd.testing.assert_frame_equal(data, loaded_data)

    @patch('data.fetchers.yahoo.yf.download')
    def test_update_data_file(self, mock_download):
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', end='2020-01-10'),
            'Close': range(10)
        }).set_index('Date')
        mock_download.return_value = mock_data

        save_data_to_file(mock_data, 'AAPL', 'csv')
        update_data_file('AAPL', 'csv')
        folder_path = os.path.join('data_store', 'AAPL')
        filename = os.path.join(folder_path, f"{date.today().strftime('%Y-%m-%d')}.csv")
        updated_data = pd.read_csv(filename, index_col=0, parse_dates=True)
        self.assertFalse(updated_data.empty)

    @patch('data.fetchers.yahoo.os.path.exists')
    @patch('data.fetchers.yahoo.os.listdir')
    @patch('data.fetchers.yahoo.pd.read_csv')
    def test_check_existing_data_csv(self, mock_read_csv, mock_listdir, mock_exists):
        mock_exists.return_value = True
        mock_listdir.return_value = ['2020-01-01.csv']
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', end='2020-01-10'),
            'Close': range(10)
        }).set_index('Date')
        mock_read_csv.return_value = mock_data

        existing_data = check_existing_data('AAPL', '2020-01-01', '2020-01-10', '1d', 'csv')
        self.assertIsInstance(existing_data, pd.DataFrame)
        self.assertFalse(existing_data.empty)

    @patch('data.fetchers.yahoo.os.path.exists')
    @patch('data.fetchers.yahoo.os.listdir')
    @patch('data.fetchers.yahoo.pd.read_parquet')
    def test_check_existing_data_parquet(self, mock_read_parquet, mock_listdir, mock_exists):
        mock_exists.return_value = True
        mock_listdir.return_value = ['2020-01-01.parquet']
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', end='2020-01-10'),
            'Close': range(10)
        }).set_index('Date')
        mock_read_parquet.return_value = mock_data

        existing_data = check_existing_data('AAPL', '2020-01-01', '2020-01-10', '1d', 'parquet')
        self.assertIsInstance(existing_data, pd.DataFrame)
        self.assertFalse(existing_data.empty)


if __name__ == '__main__':
    unittest.main()
