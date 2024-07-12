from data.fetchers.yahoo import download_multiple_tickers
from utils.calculations import n_day_high_in_last_k_days


def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2020-01-01'
    end_date = '2020-12-31'
    n = 20
    k = 5

    # Download historical data for the tickers
    data = download_multiple_tickers(tickers, start_date, end_date)

    # Check how many tickers saw an N-day high in the past k days
    for ticker in tickers:
        ticker_data = data['Close'][ticker]
        result = n_day_high_in_last_k_days(ticker_data.values, n, k)
        count = result.sum()
        print(f"{ticker} saw an N-day high {count} times in the past {k} days.")


if __name__ == "__main__":
    main()
