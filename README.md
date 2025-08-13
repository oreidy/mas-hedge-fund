# MAS Hedge Fund

This is a proof of concept for a multi-agent system hedge fund.  The goal of this project is to explore the use of AI to make trading decisions. This project is for **educational** purposes only and is not intended for real trading or investment.

**Note for ZIP file users:** If you downloaded this project as a ZIP file, you can ignore all GitHub-specific instructions throughout this README.

This system employs several agents working together:

1. Technical Analysis Agent - Analyzes technical indicators and generates trading signals
2. Sentiment Agent - Analyzes market sentiment and generates trading signals
3. Fundamentals Agent - Analyzes fundamental data and generates trading signals
4. Valuation Agent - Calculates the intrinsic value of a stock and generates trading signals
5. Warren Buffett Agent - Uses Warren Buffett's principles to generate trading signals
6. Bill Ackman Agent - Uses Bill Ackman's principles to generate trading signals
7. Risk Management Agent - Calculates risk metrics and sets position limits
8. Equity Agent - Makes final trading decisions and generates orders for stocks
9. Macro Agent - Analyzes macroeconomic indicators and market conditions
10. Forward-Looking Agent - Analyzes future market outlook and trends
11. Fixed-Income Agent - Makes trading decisions for bond ETFs and fixed-income securities

## Workflow Architecture

### Standard Workflow
![Standard Workflow](images/standard_workflow.png)

### Optimized Workflow
![Optimized Workflow](images/optimized_workflow.png)

**Note:** The optimized workflow is activated when screening mode is enabled and all six stock analyst agents (coloured yellow) are selected.


**Note**: the system simulates trading decisions, it does not actually trade.



## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No warranties or guarantees provided
- Past performance does not indicate future results
- Creator assumes no liability for financial losses
- Consult a financial advisor for investment decisions

By using this software, you agree to use it solely for learning purposes.

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
  - [Running the Hedge Fund](#running-the-hedge-fund)
  - [Running the Backtester](#running-the-backtester)
  - [Running the LSTM Backtest](#running-the-lstm-backtest)
- [Project Structure](#project-structure)
- [License](#license)

## Setup

### Option 1: Download ZIP File
1. Download and extract the ZIP file to your desired location
2. Navigate to the mas-hedge-fund directory

### Option 2: Clone from GitHub
```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

### Install Dependencies

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

### Set up API Keys

3. Set up your environment variables:
```bash
# Create .env file for your API keys
cp .env.example .env
```

4. Set your API keys in the .env file:
```bash
# For running LLMs hosted by groq (deepseek, llama3, etc.)
# Get your Groq API key from https://groq.com/
GROQ_API_KEY=your-groq-api-key

# For running LLMs hosted by openai (gpt-4o, gpt-4o-mini, etc.)
# Get your OpenAI API key from https://platform.openai.com/
OPENAI_API_KEY=your-openai-api-key

# For running LLMs hosted by anthropic (claude-3-5-sonnet, claude-3-opus, claude-3-5-haiku)
# Get your Anthropic API key from https://anthropic.com/
ANTHROPIC_API_KEY=your-anthropic-api-key
```

**Important API Key Notes:**
- You must set `GROQ_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` for the hedge fund to work
- **For backtesting**: A premium Groq API key is recommended for running the full backtest period (~$5 cost)
- **For Groq users**: The "llama-4-scout-17b" model is recommended due to favorable token per minute and request per day limits. See [Groq rate limits documentation](https://console.groq.com/docs/rate-limits) for details.
- Financial data for AAPL, GOOGL, MSFT, NVDA, and TSLA is free and does not require an API key

### GitHub Users: Prefetch Data (Recommended)

**GitHub users only:** It's recommended to run the prefetch scripts first to properly set up the caches. If you downloaded the multi-agent system hedge fund using the zip file the data is already available in the databases.

```bash
# Navigate to the prefetch_data directory
cd src/prefetch_data

# Run prefetch scripts (choose based on your needs)
poetry run python prefetch_prices.py
poetry run python prefetch_financial_metrics.py
poetry run python prefetch_market_cap.py
poetry run python prefetch_insider_trades.py
poetry run python prefetch_bond_etfs_only.py

# For company news (requires premium API key due to rate limits)
poetry run python prefetch_company_news.py
```

**Important:** Creating the company news database requires a premium Alpha Vantage API key due to API request limits. ZIP file users can skip this as the databases are pre-populated.

**Troubleshooting:** If you encounter price-related errors (especially index out of bounds), run `prefetch_prices.py` to fix corrupted cache data.

#### First-Time Data Prefetching

**Note:** When prefetching data for the first time, the process can take several minutes and you may see error messages. This is normal and expected due to the following reasons:

1. **Tickers not yet listed** - Some tickers may not have been publicly traded during the requested time period
2. **Missing financial statements** - Financial statements may not be available for certain tickers or time periods
3. **Empty financial statement line items** - Some financial statements may exist but contain empty or null values for specific line items
4. **Different financial statement structures** - Financial companies (banks, insurance companies) have different financial statement formats and may not contain standard line items expected for other industries

It was decided not to suppress errors and warnings for the sake of transparency. The prefetching process will continue and successfully cache all available data.

### File Descriptor Limit (Precautionary Measure)

As a precautionary measure, it's recommended to increase the file descriptor limit before running backtests or data processing operations. This prevents potential file descriptor limit errors when the operating system reduces available descriptors during intensive operations.

```bash
# Check current limit
ulimit -n

# Increase limit for current terminal session (run this in every new terminal)
ulimit -n 4096
```

**Note:** This change is temporary and only applies to the current terminal session. You should run `ulimit -n 4096` in every new terminal session before running the hedge fund or backtesting scripts.

## Usage

### Running the Hedge Fund

The hedge fund analyzes stocks and provides investment recommendations for a specific date (defaults to today).

```bash
# Analyze specific tickers for today
poetry run python src/main.py --tickers AAPL,MSFT,NVDA

# Analyze for a specific date
poetry run python src/main.py --tickers AAPL,MSFT,NVDA --end-date 2024-08-10

# Screen all S&P 500 tickers
poetry run python src/main.py --screen

# Screen all S&P 500 stocks with custom cash amount
poetry run python src/main.py --tickers WMT --initial-cash 50000 --show-reasoning

# Debug mode with detailed output
poetry run python src/main.py --tickers TSLA,GOOGL --debug --verbose-data
```


#### Complete CLI Options for main.py
```bash
poetry run python src/main.py [OPTIONS]

Options:
  --tickers TICKERS             Comma-separated list of stock ticker symbols
  --screen                      Screen all S&P 500 tickers instead of specifying individual tickers
  --end-date END_DATE           Analysis date for investment recommendations (YYYY-MM-DD). Defaults to today
  --initial-cash INITIAL_CASH   Initial cash position (default: 100000.0)
  --margin-requirement MARGIN   Initial margin requirement (default: 0.0)
  --show-reasoning              Show reasoning from each agent
  --show-agent-graph            Show the agent graph
  --debug                       Enable debug mode with detailed logging
  --verbose-data                Show detailed data output (works with debug mode)
```


### Running the Backtester

#### Basic Usage
```bash
# Thesis results reproduction command (used to generate and gather results)
poetry run python src/backtester.py --screen --start-date 2022-04-01 --end-date 2025-06-30 --export-returns

# Backtest specific tickers (the default is backtesting the last month)
poetry run python src/backtester.py --tickers AAPL,MSFT,NVDA

# Backtest all S&P 500 tickers
poetry run python src/backtester.py --screen

# Basic backtest with return export and LSTM training data collection
poetry run python src/backtester.py --tickers AAPL,MSFT,NVDA --export-returns --collect-training-data --training-data-format json

# Backtest specific period with custom settings
poetry run python src/backtester.py --tickers AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 --initial-capital 50000

# Full backtest with all options
poetry run python src/backtester.py --tickers AAPL,MSFT,GOOGL,TSLA --start-date 2023-01-01 --end-date 2024-12-31 --initial-capital 100000 --max-positions 20 --transaction-cost 0.001 --export-returns --debug

```

#### Complete CLI Options for backtester.py
```bash
poetry run python src/backtester.py [OPTIONS]

Options:
  --tickers TICKERS                    Comma-separated list of stock ticker symbols
  --screen                             Screen all S&P 500 tickers instead of specifying individual tickers
  --start-date START_DATE              Start date in YYYY-MM-DD format
  --end-date END_DATE                  End date in YYYY-MM-DD format
  --initial-capital INITIAL_CAPITAL    Initial capital amount (default: 100000)
  --margin-requirement MARGIN          Margin ratio for short positions (default: 0.0)
  --max-positions MAX_POSITIONS        Maximum number of active positions allowed (default: 30)
  --transaction-cost TRANSACTION_COST  Transaction cost as percentage of trade value (default: 0.0005)
  --debug                              Enable debug mode with detailed logging
  --verbose-data                       Show detailed data output (works with debug mode)
  --export-returns                     Export daily returns to CSV file in benchmarking/data/ folder
  --collect-training-data              Collect training data for RL model training
  --training-data-dir DIR              Directory to store training episode data (default: training_data)
  --training-data-format FORMAT        Format for storing training data: json or pickle (default: json)
```


### Running the LSTM Backtest

The LSTM backtest uses pre-trained machine learning models to make trading decisions based on historical agent consensus data.

#### Prerequisites
- Training data must be collected first using the regular backtester with `--collect-training-data`
- LSTM models must be pre-trained using the walk-forward training system

#### Usage
```bash
poetry run python src/rl/backtesting/lstm_backtest.py
```

**Note**: The LSTM backtest currently uses hardcoded parameters (100k initial capital, 250 initial training episodes, 30 max positions, 0.0005 transaction cost). This ensures comparability to the backtest that ran with the Equity agent that used a LLM. To customize these parameters, you would need to modify the values in the `main()` function of the script.

#### Example
```bash
# Run LSTM backtest with default parameters
poetry run python src/rl/backtesting/lstm_backtest.py
```

## Project Structure 
```
mas-hedge-fund/
├── src/
│   ├── agents/                   # Agent definitions and workflow
│   │   ├── bill_ackman.py        # Bill Ackman agent
│   │   ├── fundamentals.py       # Fundamental analysis agent
│   │   ├── equity_agent.py       # Equity trading agent
│   │   ├── fixed_income.py       # Fixed-income trading agent
│   │   ├── forward_looking.py    # Forward-looking analysis agent
│   │   ├── macro.py              # Macro analysis agent
│   │   ├── risk_manager.py       # Risk management agent
│   │   ├── sentiment.py          # Sentiment analysis agent
│   │   ├── technicals.py         # Technical analysis agent
│   │   ├── valuation.py          # Valuation analysis agent
│   │   ├── warren_buffett.py     # Warren Buffett agent
│   ├── prefetch_data/            # Data prefetching scripts
│   │   ├── prefetch_prices.py    # Prefetch stock prices
│   │   ├── prefetch_company_news.py # Prefetch company news
│   │   ├── prefetch_financial_metrics.py # Prefetch financial metrics
│   │   ├── prefetch_market_cap.py # Prefetch market cap data
│   │   ├── prefetch_insider_trades.py # Prefetch insider trades
│   │   ├── prefetch_bond_etfs_only.py # Prefetch bond ETF data
│   ├── rl/                       # LSTM and reinforcement learning
│   │   ├── backtesting/          # LSTM backtesting system
│   │   ├── models/               # LSTM model definitions
│   │   ├── training/             # Training utilities
│   │   ├── trained_models/       # Pre-trained LSTM models
│   ├── tools/                    # Agent tools
│   │   ├── api.py                # API tools
│   ├── backtester.py             # Main backtesting system
│   ├── main.py                   # Main entry point
├── run_lstm_backtest.py          # LSTM backtest runner
├── train_walkforward_models.py   # LSTM model training
├── pyproject.toml
├── ...
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
