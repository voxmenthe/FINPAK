from data.fetchers.yahoo import download_multiple_tickers
from utils.calculations import n_day_high_in_last_k_days


def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2024-05-01'
    end_date = '2024-07-31'
    n = 60
    k = 12

    # Download historical data for the tickers
    data = download_multiple_tickers(tickers, start_date, end_date)

    # Check how many tickers saw an N-day high in the past k days
    for ticker in tickers:
        ticker_data = data['Close'][ticker]
        if len(ticker_data) >= n:
            result = n_day_high_in_last_k_days(ticker_data.values, n, k)
            count = result[-k:].sum()
            print(f"{ticker} saw an N-day high {count} times in the past {k} days.")
        else:
            print(f"{ticker} does not have enough data to calculate {n}-day highs.")


if __name__ == "__main__":
    main()
