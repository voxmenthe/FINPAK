# Tests

## Overview

This directory contains all the unit tests for the Financial Analysis Tool. The tests are organized in subdirectories that mirror the structure of the `src/financial_analysis` directory.

## Directory Structure

```
tests/
├── financial_analysis/
│   ├── data/
│   │   ├── fetchers/
│   │   │   ├── test_yahoo.py
│   │   │   └── __init__.py
│   │   ├── cleaners/
│   │   │   ├── test_data_cleaner.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── analysis/
│   │   ├── indicators/
│   │   │   ├── technical/
│   │   │   │   ├── test_momentum.py
│   │   │   │   ├── test_trend.py
│   │   │   │   ├── test_volatility.py
│   │   │   │   └── __init__.py
│   │   │   ├── fundamental/
│   │   │   │   └── __init__.py
│   │   │   └── __init__.py
│   │   ├── screeners/
│   │   │   ├── test_equity_screener.py
│   │   │   ├── test_currency_screener.py
│   │   │   ├── test_screener_base.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── trading/
│   │   ├── systems/
│   │   │   ├── test_trend_following.py
│   │   │   ├── test_mean_reversion.py
│   │   │   ├── test_custom_system.py
│   │   │   └── __init__.py
│   │   ├── signals/
│   │   │   ├── test_signal_generator.py
│   │   │   └── __init__.py
│   │   ├── allocation/
│   │   │   ├── test_position_sizer.py
│   │   │   ├── test_portfolio_manager.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── performance/
│   │   ├── metrics/
│   │   │   ├── test_returns.py
│   │   │   ├── test_risk.py
│   │   │   ├── test_ratios.py
│   │   │   └── __init__.py
│   │   ├── tracking/
│   │   │   ├── test_trade_tracker.py
│   │   │   ├── test_system_tracker.py
│   │   │   └── __init__.py
│   │   ├── reporting/
│   │   │   ├── test_performance_report.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── visualization/
│   │   ├── charting/
│   │   │   ├── test_single_security.py
│   │   │   ├── test_multi_security.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── test_constants.py
│   │   ├── test_calculations.py
│   │   ├── test_aggregations.py
│   │   └── __init__.py
│   └── __init__.py
├── config/
└── README.md
```

## Running Tests

To run all tests, you can use the following command from the root directory of the project:

```bash
python -m unittest discover -s tests
```

This command will discover and run all the test cases in the `tests` directory.

Alternatively, if you are using `pytest`, you can run:

```bash
pytest tests
```

This will also discover and run all the test cases in the `tests` directory.

## Adding New Tests

When adding new tests, please follow the existing directory structure and naming conventions. Each test file should start with `test_` and should be placed in the appropriate subdirectory.

For example, if you are adding tests for a new module in `financial_analysis.data.fetchers`, you should create a new test file in `tests/financial_analysis/data/fetchers/` and name it `test_new_module.py`.

## Test Coverage

To check the test coverage, you can use the `coverage` package. First, install it using `pip`:

```bash
pip install coverage
```

Then, run the tests with coverage:

```bash
coverage run -m unittest discover -s tests
```

To generate a coverage report, use:

```bash
coverage report -m
```

To generate an HTML report, use:

```bash
coverage html
```

The HTML report will be generated in the `htmlcov` directory.
