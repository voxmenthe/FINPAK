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
    Download historical data for a single ticker, updating existing data if necessary.

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
    ticker_folder_path = os.path.join(folder_path, ticker)
    if not os.path.exists(ticker_folder_path):
        os.makedirs(ticker_folder_path)

    # check_existing_data expects the base folder (e.g. data_store), as it appends ticker itself
    existing_data_info = check_existing_data(ticker, interval, file_format, folder_path)
    
    if existing_data_info:
        existing_start = existing_data_info['start_date']
        existing_end = existing_data_info['end_date']
        
        # Load existing data
        existing_data = pd.DataFrame()
        for file_name in os.listdir(ticker_folder_path):
            file_path = os.path.join(ticker_folder_path, file_name)
            if file_format == 'csv' and file_name.endswith('.csv'):
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            elif file_format == 'parquet' and file_name.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            else:
                continue
            existing_data = pd.concat([existing_data, data])
        
        existing_data = existing_data.sort_index().drop_duplicates()
        
        new_data_parts = []
        
        # Download data before existing start date if needed
        if start_date < existing_start:
            early_data = yf.download(ticker, start=start_date, end=existing_start, interval=interval)
            new_data_parts.append(early_data)
        
        # Download data after existing end date if needed
        if end_date > existing_end:
            late_data = yf.download(ticker, start=existing_end, end=end_date, interval=interval)
            new_data_parts.append(late_data)
        
        # Combine all data
        if new_data_parts:
            updated_data = pd.concat([existing_data] + new_data_parts).sort_index().drop_duplicates()
        else:
            updated_data = existing_data
    else:
        updated_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
    if not updated_data.empty:
        save_data_to_file(updated_data, ticker, file_format, ticker_folder_path)
    
    return updated_data.loc[start_date:end_date]


#########
# Not sure if the `download_yahoo_data` function is used anywhere
def download_yahoo_data(symbol, start_date, end_date):
    existing_data = check_existing_data(symbol)
    
    if existing_data['exists']:
        if existing_data['start_date'] <= start_date and existing_data['end_date'] >= end_date:
            print(f"Data for {symbol} already exists for the specified date range.")
            return
        
        if existing_data['start_date'] > start_date:
            # Download data before existing start date
            download_and_append_data(symbol, start_date, existing_data['start_date'] - timedelta(days=1), 'prepend')
        
        if existing_data['end_date'] < end_date:
            # Download data after existing end date
            download_and_append_data(symbol, existing_data['end_date'] + timedelta(days=1), end_date, 'append')
    else:
        # Download full date range
        download_and_append_data(symbol, start_date, end_date, 'write')



def download_multiple_tickers(tickers, start_date, end_date, interval='1d', auto_adjust=False):
    """
    Download historical data for multiple tickers.

    Parameters:
    tickers (list): A list of ticker symbols.
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.
    interval (str): The data interval (e.g., '1d', '1wk', '1mo').
    auto_adjust (bool): Whether to return auto-adjusted prices.

    Returns:
    pd.DataFrame: The historical data.
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=auto_adjust
    )
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
