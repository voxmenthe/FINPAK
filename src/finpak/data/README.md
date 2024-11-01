# Data Module

## Overview

The `data` module is responsible for fetching, cleaning, and preprocessing financial data from various sources. This module ensures that the data is ready for analysis, trading, and performance evaluation.

## Submodules

### 1. Fetchers
- **yahoo.py**: Contains functions to download historical market data from Yahoo Finance using the `yfinance` library.

### 2. Cleaners
- **data_cleaner.py**: Provides functions to clean and preprocess raw financial data.

## Functions in `yahoo.py`

### `download_historical_data(ticker, start_date, end_date, interval='1d')`
Download historical data for a single ticker.

**Parameters**:
- `ticker` (str): The ticker symbol.
- `start_date` (str): The start date in 'YYYY-MM-DD' format.
- `end_date` (str): The end date in 'YYYY-MM-DD' format.
- `interval` (str): The data interval (e.g., '1d', '1wk', '1mo').

**Returns**:
- `pd.DataFrame`: The historical data.

### `download_multiple_tickers(tickers, start_date, end_date, interval='1d')`
Download historical data for multiple tickers.

**Parameters**:
- `tickers` (list): A list of ticker symbols.
- `start_date` (str): The start date in 'YYYY-MM-DD' format.
- `end_date` (str): The end date in 'YYYY-MM-DD' format.
- `interval` (str): The data interval (e.g., '1d', '1wk', '1mo').

**Returns**:
- `pd.DataFrame`: The historical data.

### `save_data_to_file(data, filename, file_format='csv')`
Save data to a file.

**Parameters**:
- `data` (pd.DataFrame): The data to save.
- `filename` (str): The name of the file.
- `file_format` (str): The file format ('csv' or 'parquet').

### `update_data_file(ticker, filename, file_format='csv')`
Update an existing data file with new data.

**Parameters**:
- `ticker` (str): The ticker symbol.
- `filename` (str): The name of the file.
- `file_format` (str): The file format ('csv' or 'parquet').

## Usage

To use the functions in `yahoo.py`, import them as follows:

```python
from financial_analysis.data.fetchers.yahoo import download_historical_data, download_multiple_tickers, save_data_to_file, update_data_file

# Example usage
data = download_historical_data('AAPL', '2020-01-01', '2021-01-01')
save_data_to_file(data, 'aapl_data.csv', 'csv')
```
