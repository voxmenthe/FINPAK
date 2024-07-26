# Tests

## Overview

This directory contains all the unit tests for the Financial Analysis Tool. The tests are organized in subdirectories that mirror the structure of the `src/financial_analysis` directory.


## Running Tests

To run all tests, you can use the following command from the root directory of the project:

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
