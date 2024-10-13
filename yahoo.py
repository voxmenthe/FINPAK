import datetime
import yfinance as yf
import pandas as pd
from datetime import timedelta

def download_and_append_data(symbol, start_date, end_date, mode):
    data = yf.download(symbol, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.date

    if mode == 'write':
        data.to_csv(f'data/{symbol}.csv', sep=',', header=True, index=False)
    elif mode == 'append':
        existing_data = pd.read_csv(f'data/{symbol}.csv', sep=',')
        combined_data = pd.concat([existing_data, data], ignore_index=True)
        combined_data.to_csv(f'data/{symbol}.csv', sep=',', header=True, index=False)
    elif mode == 'prepend':
        existing_data = pd.read_csv(f'data/{symbol}.csv', sep=',')
        combined_data = pd.concat([data, existing_data], ignore_index=True)
        combined_data.to_csv(f'data/{symbol}.csv', sep=',', header=True, index=False)

def check_existing_data(symbol):
    try:
        with open(f'data/{symbol}.csv', 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                start_date = datetime.strptime(lines[1].split(',')[0], '%Y-%m-%d').date()
                end_date = datetime.strptime(lines[-1].split(',')[0], '%Y-%m-%d').date()
                return {'exists': True, 'start_date': start_date, 'end_date': end_date}
    except FileNotFoundError:
        pass
    return {'exists': False, 'start_date': None, 'end_date': None}

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

if __name__ == "__main__":
    download_yahoo_data('AAPL', datetime.date(2020, 1, 1), datetime.date(2022, 12, 31))