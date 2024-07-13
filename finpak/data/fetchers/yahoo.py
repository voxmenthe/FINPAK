# flake8: noqa E501
import os
import pandas as pd
import yfinance as yf
from datetime import date


def check_existing_data(ticker, start_date, end_date, interval='1d', file_format='csv'):
    """
    Check if the data for the given date range exists in the data store.

    Parameters:
    ticker (str): The ticker symbol.
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.
    interval (str): The data interval (e.g., '1d', '1wk', '1mo').
    file_format (str): The file format ('csv' or 'parquet').

    Returns:
    pd.DataFrame or None: The existing data if found, otherwise None.
    """
    folder_path = os.path.join('data_store', ticker)
    if not os.path.exists(folder_path):
        return None

    data_frames = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_format == 'csv' and file_name.endswith('.csv'):
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        elif file_format == 'parquet' and file_name.endswith('.parquet'):
            data = pd.read_parquet(file_path)
        else:
            continue

        data_frames.append(data)

    if data_frames:
        all_data = pd.concat(data_frames).drop_duplicates()
        mask = (all_data.index >= start_date) & (all_data.index <= end_date)
        return all_data.loc[mask]
    return None


def download_historical_data(ticker, start_date, end_date, interval='1d', file_format='csv'):
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
    existing_data = check_existing_data(ticker, start_date, end_date, interval, file_format)
    if existing_data is not None and not existing_data.empty:
        return existing_data

    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if not data.empty:
        save_data_to_file(data, ticker, file_format)
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


def save_data_to_file(data, ticker, file_format='csv'):
    """
    Save data to a file.

    Parameters:
    data (pd.DataFrame): The data to save.
    filename (str): The name of the file.
    file_format (str): The file format ('csv' or 'parquet').
    """
    folder_path = os.path.join('data_store', ticker)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filename = os.path.join(folder_path, f"{date.today().strftime('%Y-%m-%d')}.{file_format}")

    if file_format == 'csv':
        data.to_csv(filename)
    elif file_format == 'parquet':
        data.to_parquet(filename)
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'parquet'.")


def update_data_file(ticker, file_format='csv'):
    """
    Update an existing data file with new data.

    Parameters:
    ticker (str): The ticker symbol.
    filename (str): The name of the file.
    file_format (str): The file format ('csv' or 'parquet').
    """
    folder_path = os.path.join('data_store', ticker)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    existing_data = check_existing_data(ticker, '1900-01-01', date.today().strftime('%Y-%m-%d'), file_format=file_format)
    if existing_data is not None and not existing_data.empty:
        last_date = existing_data.index[-1]
        new_data = download_historical_data(ticker, start_date=last_date.strftime('%Y-%m-%d'), end_date=date.today().strftime('%Y-%m-%d'), file_format=file_format)
        updated_data = pd.concat([existing_data, new_data]).drop_duplicates()
        if not updated_data.empty:
            save_data_to_file(updated_data, ticker, file_format)
    else:
        new_data = download_historical_data(ticker, start_date='1900-01-01', end_date=date.today().strftime('%Y-%m-%d'), file_format=file_format)
        if not new_data.empty:
            save_data_to_file(new_data, ticker, file_format)

