import pandas as pd
import yfinance as yf
from datetime import date


def download_historical_data(ticker, start_date, end_date, interval='1d'):
    """
    Download historical data for a single ticker.

    Parameters:
    ticker (str): The ticker symbol.
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.
    interval (str): The data interval (e.g., '1d', '1wk', '1mo').

    Returns:
    pd.DataFrame: The historical data.
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data


def download_multiple_tickers(tickers, start_date, end_date, interval='1d'):
    """
    Download historical data for multiple tickers.

    Parameters:
    tickers (list): A list of ticker symbols.
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.
    interval (str): The data interval (e.g., '1d', '1wk', '1mo').

    Returns:
    pd.DataFrame: The historical data.
    """
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval)
    return data


def save_data_to_file(data, filename, file_format='csv'):
    """
    Save data to a file.

    Parameters:
    data (pd.DataFrame): The data to save.
    filename (str): The name of the file.
    file_format (str): The file format ('csv' or 'parquet').
    """
    if file_format == 'csv':
        data.to_csv(filename)
    elif file_format == 'parquet':
        data.to_parquet(filename)
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'parquet'.")


def update_data_file(ticker, filename, file_format='csv'):
    """
    Update an existing data file with new data.

    Parameters:
    ticker (str): The ticker symbol.
    filename (str): The name of the file.
    file_format (str): The file format ('csv' or 'parquet').
    """
    if file_format == 'csv':
        existing_data = pd.read_csv(filename, index_col=0, parse_dates=True)
    elif file_format == 'parquet':
        existing_data = pd.read_parquet(filename)
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'parquet'.")

    last_date = existing_data.index[-1]
    new_data = download_historical_data(ticker, start_date=last_date.strftime('%Y-%m-%d'), end_date=date.today().strftime('%Y-%m-%d'))
    updated_data = pd.concat([existing_data, new_data]).drop_duplicates()

    save_data_to_file(updated_data, filename, file_format)

