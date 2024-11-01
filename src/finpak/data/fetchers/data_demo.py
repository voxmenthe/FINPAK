import os
from yahoo import check_existing_data, update_data_file, list_available_data

def main():
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    file_format = 'csv'

    # List available data
    print("Available data files:")
    available_files = list_available_data(ticker, file_format)
    for file in available_files:
        print(file)

    # Check existing data
    print("\nChecking existing data:")
    existing_data = check_existing_data(ticker, start_date, end_date, file_format=file_format)
    if existing_data is not None:
        print(existing_data)
    else:
        print("No existing data found for the specified range.")

    # Update data file
    print("\nUpdating data file:")
    update_data_file(ticker, file_format=file_format)
    print("Data file updated.")

if __name__ == "__main__":
    main()
