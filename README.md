
# Financial Analysis Tool

## Overview

This Python package provides a comprehensive suite of tools for financial analysis, focusing on equities and currencies. It combines data fetching, technical and fundamental analysis, screening, trading systems, and performance tracking into a single, modular framework.

## Key Features

- **Data Handling**: Fetch and clean financial data from various sources, including Yahoo Finance.
- **Technical and Fundamental Analysis**: Calculate a wide range of indicators for in-depth market analysis.
- **Screening**: Implement custom screening criteria for equities and currencies.
- **Trading Systems**: Generate buy/sell signals and manage portfolio allocation.
- **Performance Tracking**: Monitor and analyze the performance of trading strategies.
- **Visualization**: Create insightful charts for single or multiple securities.

## Module Structure

### 1. Data Module (`financial_analysis.data`)
- Fetches raw financial data from various sources.
- Cleans and preprocesses data for analysis.

### 2. Analysis Module (`financial_analysis.analysis`)
- **Indicators**: Calculates technical and fundamental indicators.
- **Screeners**: Filters securities based on user-defined criteria.

### 3. Trading Module (`financial_analysis.trading`)
- **Systems**: Implements various trading strategies (e.g., trend following, mean reversion).
- **Signals**: Generates buy/sell signals based on analysis results.
- **Allocation**: Manages position sizing and portfolio composition.

### 4. Performance Module (`financial_analysis.performance`)
- **Metrics**: Calculates performance metrics (returns, risk measures, ratios).
- **Tracking**: Monitors individual trades and overall system performance.
- **Reporting**: Generates comprehensive performance reports.

### 5. Visualization Module (`financial_analysis.visualization`)
- Creates charts for technical indicators and price data.
- Supports both single and multi-security visualizations.

### 6. Utilities (`financial_analysis.utils`)
- Provides common functions for calculations and data aggregation used across modules.

## Workflow

1. **Data Acquisition**: The data module fetches and prepares financial data.
2. **Analysis**: Indicators are calculated, and securities are screened based on user criteria.
3. **Trading Decisions**: Trading systems use the analysis results to generate signals and make allocation decisions.
4. **Performance Evaluation**: The performance of trades and overall strategies is tracked and analyzed.
5. **Visualization**: Results can be visualized at various stages for deeper insights.

---

This project aims to provide a flexible and powerful toolkit for financial analysis and trading strategy development. By leveraging the modular structure, users can easily extend or customize functionality to suit their specific needs.


### Repo Directory Structure
FINPAK/
├── finpak/
|   ├── pipelines/
│   ├── data/
│   │   ├── fetchers/
│   │   │   ├── yahoo.py
│   │   │   └── __init__.py
│   │   ├── cleaners/
│   │   │   ├── data_cleaner.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── analysis/
│   │   ├── indicators/
│   │   │   ├── technical/
│   │   │   │   ├── momentum.py
│   │   │   │   ├── trend.py
│   │   │   │   ├── volatility.py
│   │   │   │   └── __init__.py
│   │   │   ├── fundamental/
│   │   │   │   └── __init__.py
│   │   │   └── __init__.py
│   │   ├── screeners/
│   │   │   ├── equity_screener.py
│   │   │   ├── currency_screener.py
│   │   │   ├── screener_base.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── trading/
│   │   ├── systems/
│   │   │   ├── trend_following.py
│   │   │   ├── mean_reversion.py
│   │   │   ├── custom_system.py
│   │   │   └── __init__.py
│   │   ├── signals/
│   │   │   ├── signal_generator.py
│   │   │   └── __init__.py
│   │   ├── allocation/
│   │   │   ├── position_sizer.py
│   │   │   ├── portfolio_manager.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── performance/
│   │   ├── metrics/
│   │   │   ├── returns.py
│   │   │   ├── risk.py
│   │   │   ├── ratios.py
│   │   │   └── __init__.py
│   │   ├── tracking/
│   │   │   ├── trade_tracker.py
│   │   │   ├── system_tracker.py
│   │   │   └── __init__.py
│   │   ├── reporting/
│   │   │   ├── performance_report.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── visualization/
│   │   ├── charting/
│   │   │   ├── single_security.py
│   │   │   ├── multi_security.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── utils/
|   |   |── constants.py
│   │   ├── calculations.py
│   │   ├── aggregations.py
│   │   └── __init__.py
│   └── __init__.py
├── tests/
│   └── financial_analysis/
│       ├── data/
│       ├── analysis/
│       ├── trading/
│       ├── performance/
│       ├── visualization/
│       └── utils/
├── config/
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md