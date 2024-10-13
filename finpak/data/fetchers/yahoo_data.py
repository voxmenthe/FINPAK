# flake8: noqa E501
import os
import pandas as pd
import yfinance as yf
from datetime import date


def list_available_data(ticker, file_format='csv', folder_path='data_store'):
    """
    List available data files for a given ticker and show data ranges.

    Parameters:
    ticker (str): The ticker symbol.
    file_format (str): The file format ('csv' or 'parquet').
    folder_path (str): The folder path where data is stored.

    Returns:
    dict: A dictionary with file names as keys and date ranges as values.
    """
    folder_path = os.path.join(folder_path, ticker)
    if not os.path.exists(folder_path):
        return {}

    available_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(f'.{file_format}'):
            file_path = os.path.join(folder_path, file_name)
            if file_format == 'csv':
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            elif file_format == 'parquet':
                data = pd.read_parquet(file_path)
            else:
                continue
            
            date_range = f"{data.index.min().date()} to {data.index.max().date()}"
            available_data[file_name] = date_range

    return available_data


def check_existing_data(ticker, interval='1d', file_format='csv', folder_path='data_store'):
    """
    Check if the data for the given date range exists in the data store.

    Parameters:
    ticker (str): The ticker symbol.
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.
    interval (str): The data interval (e.g., '1d', '1wk', '1mo').
    file_format (str): The file format ('csv' or 'parquet').

    Returns:
    dict: A dictionary with 'start_date' and 'end_date' of existing data, or an empty dict if no data found.
    """
    folder_path = os.path.join(folder_path, ticker)
    if not os.path.exists(folder_path):
        return {}

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
        all_data = pd.concat(data_frames).drop_duplicates().sort_index()
        return {
            'start_date': all_data.index.min().strftime('%Y-%m-%d'),
            'end_date': all_data.index.max().strftime('%Y-%m-%d')
        }
    return {}


def download_historical_data(ticker, start_date, end_date, interval='1d', file_format='csv', folder_path='data_store'):
    """
    Download historical data for a single ticker, only downloading additional data if it doesn't exist.

    Parameters:
    ticker (str): The ticker symbol.
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.
    interval (str): The data interval (e.g., '1d', '1wk', '1mo').
    file_format (str): The file format ('csv' or 'parquet').
    folder_path (str): The folder path where data is stored.

    Returns:
    pd.DataFrame: The historical data.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    ticker_path = os.path.join(folder_path, ticker) + f".{file_format}"

    existing_data_info = check_existing_data(ticker, interval, file_format, folder_path)        

    if existing_data_info:
        existing_start = pd.to_datetime(existing_data_info['start_date'])
        existing_end = pd.to_datetime(existing_data_info['end_date'])
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Check if we need to download additional data
        need_earlier_data = start_date < existing_start
        need_later_data = end_date > existing_end

        if need_earlier_data or need_later_data:
            # Load existing data
            if file_format == 'csv' and ticker_path.endswith('.csv'):
                existing_data = pd.read_csv(ticker_path, index_col=0, parse_dates=True)
            elif file_format == 'parquet' and ticker_path.endswith('.parquet'):
                existing_data = pd.read_parquet(ticker_path)
            
            existing_data = existing_data.sort_index().drop_duplicates()

            if need_earlier_data:
                earlier_data = yf.download(ticker, start=start_date, end=existing_start, interval=interval)
                existing_data = pd.concat([earlier_data, existing_data])
            
            if need_later_data:
                later_data = yf.download(ticker, start=existing_end, end=end_date, interval=interval)
                existing_data = pd.concat([existing_data, later_data])
            
            updated_data = existing_data.sort_index().drop_duplicates()
        else:
            updated_data = existing_data.sort_index().drop_duplicates()
    else:
        updated_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
    if not updated_data.empty:
        save_data_to_file(updated_data, ticker, file_format, folder_path)
    
    return updated_data.loc[start_date:end_date]


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


def save_data_to_file(data, ticker, file_format='csv', folder_path='data_store'):
    """
    Save data to a file.

    Parameters:
    data (pd.DataFrame): The data to save.
    filename (str): The name of the file.
    file_format (str): The file format ('csv' or 'parquet').
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filename = os.path.join(folder_path, f"{ticker}.{file_format}")

    if file_format == 'csv':
        data.to_csv(filename)
    elif file_format == 'parquet':
        data.to_parquet(filename)
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'parquet'.")