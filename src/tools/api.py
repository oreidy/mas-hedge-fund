import os
import pandas as pd
import yfinance as yf
import asyncio
import time
# import aiohttp  # No longer needed
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import traceback
import requests

from data.disk_cache import DiskCache
from data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
)

# Import other functions
from utils.market_calendar import is_trading_day, adjust_date_range, adjust_yfinance_date_range, get_previous_trading_day, get_next_trading_day
from tools.sec_edgar_scraper import fetch_multiple_insider_trades
from data.company_news_db import (
    save_company_news,
    get_company_news as get_company_news_from_db,
    get_multiple_company_news,
    is_news_cache_valid
)
from utils.logger import logger


def get_monthly_cache_period(end_date: str) -> str:
    """Convert end_date to monthly cache period (YYYY-MM format)"""
    from datetime import datetime
    date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    return f"{date_obj.year}-{date_obj.month:02d}"


# ===== CACHE INITIALIZATION =====
# Create a global cache instance with disk persistence
# Use absolute path to ensure consistent cache location regardless of execution directory
_cache = DiskCache(cache_dir=Path(__file__).parent.parent.parent / "cache")


# ===== PRICE FUNCTIONS =====
def get_prices(ticker: str, start_date: str, end_date: str, verbose_data: bool = False) -> List[Price]:
    """
    Fetch price data for a single ticker.
    This is the main public API that maintains compatibility with the original function.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        verbose_data: Optional flag for verbose logging
        
    Returns:
        List of Price objects
    """

    # overwritten to False for the sake of clarity
    verbose_data = False

    # Log function call
    logger.debug(f"Fetching prices for {ticker} from {start_date} to {end_date}", module="get_prices", ticker=ticker)

    # Run the async function in a new event loop
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(get_prices_async([ticker], start_date, end_date, verbose_data))
        if verbose_data:
            logger.debug(f"results: {results}", module="get_prices", ticker=ticker)
        return results.get(ticker, []) # Review: why is it not just return results? Why .get()?
    finally:
        loop.close() # Review: Why do I have to close the loop? Why can't I just omit this finally:?


async def get_prices_async(tickers: List[str], start_date: str, end_date: str, verbose_data: bool = False) -> Dict[str, List[Price]]:
    """Fetch prices for multiple tickers asynchronously"""

    # overwritten to False for the sake of clarity
    verbose_data = False

    logger.debug(f"Running get_prices_async() for {len(tickers)} tickers from {start_date} to {end_date}", 
                module="get_prices_async")

    # Ensure dates are strings
    start_date = str(start_date) if not isinstance(start_date, str) else start_date
    end_date = str(end_date) if not isinstance(end_date, str) else end_date

    # Check if dates need adjustment for trading days
    logger.debug(f"Checking if {start_date} and {end_date} are trading days.", 
                        module="get_prices_async")

    if not is_trading_day(start_date) or not is_trading_day(end_date):
        try:
            if verbose_data:
                logger.debug(f"Either start_date {start_date} or end_date {end_date} isn't a trading day.", module="get_prices_async")
            start_date, end_date = adjust_date_range(start_date, end_date)
        except ValueError as e:
            logger.warning(f"Warning: {e} - continuing with original dates", module="get_prices_async")

    df_results = await fetch_prices_batch(tickers, start_date, end_date, verbose_data) # Review: Why do I need "await" here?
    if verbose_data:
        logger.debug(f"df_results: {df_results}", module="get_prices_async", ticker=tickers)
    return {ticker: df_to_price_objects(df, ticker) for ticker, df in df_results.items()}


async def fetch_prices_batch(tickers: List[str], start_date: str, end_date: str, verbose_data: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical prices for multiple tickers in a single batch request.
    Uses yfinance's batch download capability for efficiency.
    Adjusts dates to trading days to handle market holidays.
    """
    
    # overwritten to False for the sake of clarity
    verbose_data = False

    logger.debug(f"Running fetch_prices_batch() for {len(tickers)} tickers from {start_date} to {end_date}", module="fetch_prices_batch")

    # First check cache for all tickers
    cached_results = {}
    tickers_to_fetch = []
    
    # Cache tracking counters
    cache_hits = 0
    total_tickers = len(tickers)

    # Debug: cache structure
    if verbose_data:
        for ticker in tickers:  # Just check a few tickers to understand cache behavior
            cache_data = _cache.get_prices(ticker)
            if cache_data:
                # Check if the cache contains data within our date range
                date_filtered = [p for p in cache_data if start_date <= p["time"] <= end_date]
                logger.debug(f"Cache check for {ticker}: total cached items={len(cache_data)}, within date range={len(date_filtered)}", 
                           module="fetch_prices_batch", ticker=ticker)
            else:
                logger.debug(f"No cache data found for {ticker}", module="fetch_prices_batch", ticker=ticker)

    # Fetch data from cache first
    for ticker in tickers:
        cached_data = _cache.get_prices(ticker, start_date, end_date)
        if cached_data:
            # Convert cached data to DataFrame
            df = pd.DataFrame(cached_data)
            df["Date"] = pd.to_datetime(df["time"])
            df.set_index("Date", inplace=True)
            cached_results[ticker] = df
            cache_hits += 1
        else:
            tickers_to_fetch.append(ticker)

    # Debug: Log cache statistics
    if verbose_data:
        cache_percentage = (cache_hits / total_tickers * 100) if total_tickers > 0 else 0
        download_percentage = 100 - cache_percentage
        logger.debug(f"PRICE DATA CACHE STATS:", module="fetch_prices_batch")
        logger.debug(f"  - From cache: {cache_hits}/{total_tickers} tickers ({cache_percentage:.1f}%)", module="fetch_prices_batch")
        logger.debug(f"  - To download: {len(tickers_to_fetch)}/{total_tickers} tickers ({download_percentage:.1f}%)", module="fetch_prices_batch")
    


    # If there is nothing to fetch, return the cached data
    if not tickers_to_fetch:
        # Debug: Preview data when everything comes from cache
        if verbose_data and cached_results:
            example_ticker = list(cached_results.keys())[0]
            df_sample = cached_results[example_ticker]
            logger.debug(f"Debug from fetch_prices_batch if all data is in cache.",
                module="fetch_prices_batch", ticker=example_ticker)
            logger.debug(f"Sample cached price data for {example_ticker}:",
                module="fetch_prices_batch", ticker=example_ticker)
            logger.debug(f"Shape: {df_sample.shape}",
                module="fetch_prices_batch", ticker=example_ticker)
            logger.debug(f"Date range: {df_sample.index.min()} to {df_sample.index.max()}",
                module="fetch_prices_batch", ticker=example_ticker)
            logger.debug(f"First row:\n{df_sample.head(1)}",
                module="fetch_prices_batch", ticker=example_ticker)
            logger.debug(f"Last row:\n{df_sample.tail(1)}",
                module="fetch_prices_batch", ticker=example_ticker)

        return cached_results
    
    # Fetch data for tickers that aren't in cache
    try:
        # Adjust the end_date for yfinance (add one day end because in yfinance is exclusive)
        yf_start_date, yf_end_date = adjust_yfinance_date_range(start_date, end_date)

        # yfinance batch download
        logger.debug(f"Downloading price data for {len(tickers_to_fetch)} tickers", 
                         module="fetch_prices_batch")
        data = yf.download(
            tickers_to_fetch, 
            start=yf_start_date, 
            end=yf_end_date, 
            group_by='ticker',
            auto_adjust=True,
            progress=False,
            timeout=20
        )
        
        # Process the results
        results = cached_results.copy()
        
        # Debug: Structure of the downloaded data
        if isinstance(data, pd.DataFrame) and verbose_data:
            logger.debug(f"Data type: {type(data)}", module="fetch_prices_batch")
            if len(data.columns) > 0:
                logger.debug(f"Columns: {data.columns}", module="fetch_prices_batch")
                # Check if it's a MultiIndex
                if isinstance(data.columns, pd.MultiIndex):
                    logger.debug(f"MultiIndex levels: {data.columns.levels}", 
                                 module="fetch_prices_batch")
        
        # Process data as MultiIndex DataFrame
        if isinstance(data, pd.DataFrame) and not data.empty:
            # For each ticker in our fetch list
            for ticker in tickers_to_fetch:
                # Check if this ticker exists in the data
                if ticker in data.columns.levels[0]:
                    # Extract the ticker-specific data
                    ticker_data = data[ticker].copy()
                    
                    # Ensure the column names are lowercase
                    ticker_data.columns = ticker_data.columns.str.lower()
                    
                    # Check for 'adj close' and rename to 'close' if needed
                    if 'adj close' in ticker_data.columns:
                        ticker_data.rename(columns={'adj close': 'close'}, inplace=True)

                    # Cache the processed data
                    _cache.set_prices(ticker, df_to_cache_format(ticker_data))
                    results[ticker] = ticker_data
                else:
                    # Ticker not available for this date range - log warning and add empty DataFrame
                    logger.warning(f"No price data available for {ticker} during {start_date} to {end_date} (possibly not listed yet or delisted)", 
                                module="fetch_prices_batch", ticker=ticker)
                    results[ticker] = pd.DataFrame(columns=["open", "close", "high", "low", "volume"])
        else:
            # No data downloaded for any ticker - add empty DataFrames for all
            for ticker in tickers_to_fetch:
                logger.warning(f"No price data available for {ticker} during {start_date} to {end_date} (possibly not listed yet or delisted)", 
                            module="fetch_prices_batch", ticker=ticker)
                results[ticker] = pd.DataFrame(columns=["open", "close", "high", "low", "volume"])

        # Debug: Before return, preview the downloaded data
        if len(tickers_to_fetch) > 0 and len(results) > 0 and verbose_data:
            for ticker in tickers_to_fetch[:1]:  # Preview first downloaded ticker
                logger.debug(f"Debug from fetch_prices_batch if some data is downloaded.",
                        module="fetch_prices_batch", ticker=ticker)
                logger.debug(f"Sample downloaded price data for {ticker}:",
                        module="fetch_prices_batch", ticker=ticker)
                logger.debug(f"Shape: {results[ticker].shape}",
                        module="fetch_prices_batch", ticker=ticker)
                logger.debug(f"Date range: {results[ticker].index.min()} to {results[ticker].index.max()}",
                        module="fetch_prices_batch", ticker=ticker)
                logger.debug(f"First row:\n{results[ticker].head(1)}" ,
                        module="fetch_prices_batch", ticker=ticker)# Review: Doees this make sense if there is only one ticker?
                logger.debug(f"Last row:\n{results[ticker].tail(1)}",
                        module="fetch_prices_batch", ticker=ticker)

        return results
        
    except Exception as e:
        logger.error(f"Error fetching batch price data: {e}", module="fetch_prices_batch") # Review: why "as e:" ?
        return cached_results


def get_price_data(ticker: str, start_date: str, end_date: str, verbose_data: bool = False) -> pd.DataFrame:
    """Get price data as DataFrame."""

    # overwritten to False for the sake of clarity
    verbose_data = False

    logger.debug(f"Running get_price_data() for {ticker} from {start_date} to {end_date}", 
                module="get_price_data", ticker=ticker)

    prices = get_prices(ticker, start_date, end_date, verbose_data)

    df = prices_to_df(prices)
    
    if verbose_data:
        logger.debug(f"prices: {prices}", module="get_prices_data", ticker=ticker)
        logger.debug(f"df: {df}", module="get_price_data", ticker=ticker)

    return df


# ===== DATA CONVERSION UTILITIES =====
def df_to_cache_format(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to a format suitable for caching"""
    df = df.reset_index()
    df['time'] = df['Date'].dt.strftime('%Y-%m-%d')
    result = df.to_dict('records')
    return result


def df_to_price_objects(df: pd.DataFrame, ticker: str = "UNKNOWN") -> List[Price]:
    """Convert DataFrame to a list of Price objects"""
    
    # Handle MultiIndex DataFrames (flatten to regular DataFrame)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # Try to extract data for this specific ticker
            if ticker in df.columns.levels[0]:
                df = df[ticker].copy()
            else:
                # Single ticker case - flatten MultiIndex by taking price level
                df = df.copy()
                df.columns = df.columns.get_level_values(1)  # Get price types
            logger.debug(f"Handled MultiIndex DataFrame for ticker {ticker}", 
                          module="df_to_price_objects", ticker=ticker)
        except Exception as e:
            logger.debug(f"Failed to handle MultiIndex for ticker {ticker}: {e}", 
                          module="df_to_price_objects", ticker=ticker)
    
    prices = []
    for _, row in df.iterrows():
        # Handle volume conversion with fallback for NaN values
        try:
            volume = int(row['volume'])
        except (ValueError, TypeError):
            volume = 0
            logger.critical(f"Volume conversion failed for ticker {ticker} on date {row.name.strftime('%Y-%m-%d')} - value: {row['volume']}, setting to 0", 
                          module="df_to_price_objects", ticker=ticker)
        
        price = Price(
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=volume,
            time=row.name.strftime('%Y-%m-%d')
        )
        prices.append(price)
    return prices


def prices_to_df(prices: List[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame. Same as original function."""
    if not prices:
        # Return empty DataFrame with expected columns if no prices
        return pd.DataFrame(columns=["open", "close", "high", "low", "volume"])
    
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# ===== FINANCIAL METRICS FUNTIONS =====
def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "annual",
    limit: int = 10,
    verbose_data: bool = False
) -> List[FinancialMetrics]:
    """
    Fetch financial metrics for a single ticker.
    Maintains compatibility with the original function.
    
    Args:
        ticker: Stock ticker symbol
        end_date: End date string (YYYY-MM-DD)
        period: Period type ("quaterly", "annual", etc.)
        limit: Maximum number of periods to return
        verbose_data: Optional flag for verbose logging
        
    Returns:
        List of FinancialMetrics objects
    """

    logger.debug(f"Running get_financial_metrics() for {ticker} up to {end_date} (period: {period}, limit: {limit})", 
                module="get_financial_metrics", ticker=ticker)

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(fetch_financial_metrics_async([ticker], end_date, period, limit, verbose_data))
        return results.get(ticker, [])
    except Exception as e:
        logger.error(f"Error fetching financial metrics for {ticker} before {end_date}: {e}", 
                    module="get_financial_metrics", ticker=ticker)
    finally:
        loop.close()


async def fetch_financial_metrics_async(tickers: List[str], end_date: str, period: str = "annual", limit: int = 10, verbose_data: bool = False):
    """Fetch financial metrics for multiple tickers in parallel"""

    logger.debug(f"Running fetch_financial_metrics_async() for {len(tickers)} tickers", 
                module="fetch_financial_metrics_async")  

    # Counter for cache tracking
    cache_hits = 0
    total_tickers = len(tickers)
    results_dict = {}

    # Helper function that fetches financial metrics for one ticker
    async def fetch_ticker_metrics(ticker):
        nonlocal cache_hits, results_dict

        # Get monthly cache period for this request
        monthly_period = get_monthly_cache_period(end_date)
        
        cached_data = _cache.get_financial_metrics(ticker, monthly_period, period)
        if cached_data:
            filtered_data = [metric for metric in cached_data if metric["report_period"] <= end_date]
            filtered_data.sort(key=lambda x: x["report_period"], reverse=True)
            if filtered_data:
                cache_hits += 1
                metrics = [FinancialMetrics(**metric) for metric in filtered_data[:limit]]
                results_dict[ticker] = {'source': 'cache', 'data': metrics}
                return ticker, metrics
        
        # Not in cache, fetch from yfinance
        try:
            financial_metrics = []
            ticker_obj = yf.Ticker(ticker)
            
            # Get financials (income statement, balance sheet and cash flow statement)
            income_stmt = ticker_obj.income_stmt
            balance_sheet = ticker_obj.balance_sheet
            cash_flow = ticker_obj.cashflow
            
            if income_stmt.empty or balance_sheet.empty or cash_flow.empty:
                logger.warning(f"Empty financial statements for {ticker}", 
                              module="fetch_financial_metrics_async", ticker=ticker)
                return ticker, []
            
            # Get quarterly or annual data based on period
            is_quarterly = period.lower() in ["q", "quarterly", "ttm"]
            if is_quarterly:
                income_stmt = ticker_obj.quarterly_income_stmt
                balance_sheet = ticker_obj.quarterly_balance_sheet
                cash_flow = ticker_obj.quarterly_cashflow
            
            # Check if current ratio data is available once before the loop
            if 'Current Assets' not in balance_sheet.index or 'Current Liabilities' not in balance_sheet.index:
                logger.debug("Missing data for current_ratio calculation: Current Assets or Current Liabilities not found in balance sheet", 
                        module="fetch_financial_metrics_async", ticker=ticker)
            
            # Process the financial data
            for i, col in enumerate(income_stmt.columns):
                if i >= limit:
                    break
                
                report_date = col.strftime('%Y-%m-%d')
                if report_date > end_date:
                    continue
                
                # Calculate metrics from financial statements
                try:
                    total_revenue = float(income_stmt.loc['Total Revenue', col]) if 'Total Revenue' in income_stmt.index else None
                    net_income = float(income_stmt.loc['Net Income', col]) if 'Net Income' in income_stmt.index else None
                    total_assets = float(balance_sheet.loc['Total Assets', col]) if 'Total Assets' in balance_sheet.index else None
                    total_liabilities = float(balance_sheet.loc['Total Liabilities Net Minority Interest', col]) if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None
                    stockholders_equity = float(balance_sheet.loc['Stockholders Equity', col]) if 'Stockholders Equity' in balance_sheet.index else None
                    
                    # Get the market cap (with separate error handling)
                    market_cap = None
                    try:
                        market_cap = get_historical_market_cap(ticker, report_date, verbose_data=verbose_data, balance_sheet=balance_sheet, balance_sheet_column=col)
                    except Exception as market_cap_error:
                        logger.warning(f"Could not get market cap for {ticker} at {report_date}: {market_cap_error}", 
                                     module="fetch_financial_metrics_async", ticker=ticker)
                        market_cap = None
                    
                    # ROE, margins, etc.
                    return_on_equity = net_income / stockholders_equity if stockholders_equity and net_income else None
                    net_margin = net_income / total_revenue if total_revenue and net_income else None
                    operating_income = float(income_stmt.loc['Operating Income', col]) if 'Operating Income' in income_stmt.index else None
                    operating_margin = operating_income / total_revenue if total_revenue and operating_income else None
                    debt_to_equity = total_liabilities / stockholders_equity if stockholders_equity and total_liabilities else None
                    
                    # Growth metrics
                    revenue_growth = None
                    earnings_growth = None
                    book_value_growth = None

                    if i < len(income_stmt.columns) - 1:
                        prev_col = income_stmt.columns[i + 1] 
                        
                        # Revenue growth
                        prev_revenue = float(income_stmt.loc['Total Revenue', prev_col]) if 'Total Revenue' in income_stmt.index else None
                        if total_revenue and prev_revenue and prev_revenue > 0:
                            revenue_growth = (total_revenue - prev_revenue) / prev_revenue

                        if verbose_data:
                            logger.debug(f"prev_revenue: {prev_revenue}, revenue_groth: {revenue_growth}", module="fetch_financial_metrics_async")
                        
                        # Earnings growth
                        prev_net_income = float(income_stmt.loc['Net Income', prev_col]) if 'Net Income' in income_stmt.index else None
                        if net_income and prev_net_income and prev_net_income > 0:
                            earnings_growth = (net_income - prev_net_income) / prev_net_income
                        
                        if verbose_data:
                            logger.debug(f"prev_net_income: {prev_net_income}, earnings_groth: {earnings_growth}", module="fetch_financial_metrics_async")
                        
                        # Book value growth
                        prev_equity = float(balance_sheet.loc['Stockholders Equity', prev_col]) if 'Stockholders Equity' in balance_sheet.index else None
                        if stockholders_equity and prev_equity and prev_equity > 0:
                            book_value_growth = (stockholders_equity - prev_equity) / prev_equity

                        if verbose_data:
                            logger.debug(f"prev_equity: {prev_equity}, book_value_groth: {book_value_growth}", module="fetch_financial_metrics_async")
                    
                    # Free cash flow
                    operating_cash_flow = float(cash_flow.loc['Operating Cash Flow', col]) if 'Operating Cash Flow' in cash_flow.index else None
                    capital_expenditure = float(cash_flow.loc['Capital Expenditure', col]) if 'Capital Expenditure' in cash_flow.index else None
                    free_cash_flow = (operating_cash_flow + capital_expenditure) if operating_cash_flow is not None and capital_expenditure is not None else None

                    # Calculations for current_ratio
                    current_assets = float(balance_sheet.loc['Current Assets', col]) if 'Current Assets' in balance_sheet.index else None
                    current_liabilities = float(balance_sheet.loc['Current Liabilities', col]) if 'Current Liabilities' in balance_sheet.index else None
                    current_ratio = current_assets / current_liabilities if current_assets and current_liabilities else None

                    # Get shares outstanding for per-share metrics
                    shares_outstanding = float(balance_sheet.loc['Ordinary Shares Number', col]) if 'Ordinary Shares Number' in balance_sheet.index else None

                    # Calculate EPS and FCF per share
                    earnings_per_share = net_income / shares_outstanding if shares_outstanding and net_income else None
                    free_cash_flow_per_share = free_cash_flow / shares_outstanding if shares_outstanding and free_cash_flow else None
            
                    # Create FinancialMetrics object
                    metrics = FinancialMetrics(
                        ticker=ticker,
                        calendar_date=report_date,
                        report_period=report_date,
                        period=period,
                        currency="USD",
                        market_cap=market_cap,
                        price_to_earnings_ratio=market_cap / net_income if market_cap and net_income and net_income > 0 else None,
                        price_to_book_ratio=market_cap / stockholders_equity if market_cap and stockholders_equity and stockholders_equity > 0 else None,
                        price_to_sales_ratio=market_cap / total_revenue if market_cap and total_revenue and total_revenue > 0 else None,
                        enterprise_value=None,  # Would need to calculate
                        enterprise_value_to_ebitda_ratio=None, # Review: Why are some of these metrics equal to None?
                        enterprise_value_to_revenue_ratio=None,
                        free_cash_flow_yield=free_cash_flow / market_cap if market_cap and free_cash_flow else None,
                        gross_margin=float(income_stmt.loc['Gross Profit', col]) / total_revenue if 'Gross Profit' in income_stmt.index and total_revenue else None,
                        operating_margin=operating_margin,
                        net_margin=net_margin,
                        return_on_equity=return_on_equity,
                        return_on_assets=net_income / total_assets if total_assets and net_income else None,
                        return_on_invested_capital=None,  # Need more data
                        debt_to_equity=debt_to_equity,
                        current_ratio=current_ratio,  
                        revenue_growth=revenue_growth,
                        earnings_growth=earnings_growth,
                        book_value_growth=book_value_growth,
                        free_cash_flow_per_share=free_cash_flow_per_share,  
                        earnings_per_share=earnings_per_share, 
                        book_value_per_share=None, 
                        # Many fields set to None as they require additional calculation
                        **{field: None for field in [
                            'peg_ratio', 'asset_turnover', 'inventory_turnover', 'receivables_turnover',
                            'days_sales_outstanding', 'operating_cycle', 'working_capital_turnover',
                            'quick_ratio', 'cash_ratio', 'operating_cash_flow_ratio', 'debt_to_assets',
                            'interest_coverage', 'earnings_per_share_growth', 
                            'free_cash_flow_growth', 'operating_income_growth', 'ebitda_growth', 'payout_ratio'
                        ]}
                    )
                    financial_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"Error processing financial data for {ticker} at {report_date}: {e}", 
                                module="fetch_financial_metrics_async", ticker=ticker)
                    continue
            
            # Cache the results using monthly period
            if financial_metrics:
                monthly_period = get_monthly_cache_period(end_date)
                _cache.set_financial_metrics(ticker, monthly_period, period, [metric.model_dump() for metric in financial_metrics])
            
            results_dict[ticker] = {'source': 'download', 'data': financial_metrics}
            return ticker, financial_metrics
        
        except Exception as e:
            logger.error(f"Error fetching financial metrics for {ticker}: {e}", 
                        module="fetch_financial_metrics_async", ticker=ticker)
            return ticker, []
    
    # Create tasks for all tickers
    tasks = [fetch_ticker_metrics(ticker) for ticker in tickers] # Review: What are these tasks for?
    results = await asyncio.gather(*tasks) # Review: what is the * and the .gather
    
    # Debug: Log cache statistics
    if verbose_data:
        cache_percentage = (cache_hits / total_tickers * 100) if total_tickers > 0 else 0
        download_percentage = 100 - cache_percentage
        logger.debug(f"FINANCIAL METRICS CACHE STATS:", 
                    module="fetch_financial_metrics_async")
        logger.debug(f"  - From cache: {cache_hits}/{total_tickers} tickers ({cache_percentage:.1f}%)", 
                    module="fetch_financial_metrics_async")
        logger.debug(f"  - Downloaded: {total_tickers - cache_hits}/{total_tickers} tickers ({download_percentage:.1f}%)", 
                    module="fetch_financial_metrics_async")
    
    # Review: Can the following code be simplified?
    # Debug: Data preview
    if verbose_data:
        for source in ['cache', 'download']: # Review: Why does it know where the data is from?
            previewed = False # Review: what is this for?
            for ticker, info in results_dict.items():
                if info['source'] == source and info['data']:
                    logger.debug(f"Sample {source} financial metrics for {ticker}:",
                        module="fetch_financial_metrics_async", ticker=ticker)
                    logger.debug(f"Metrics count: {len(info['data'])}",
                        module="fetch_financial_metrics_async", ticker=ticker)
                    logger.debug(f"Sample metric period: {info['data'][0].report_period}",
                        module="fetch_financial_metrics_async", ticker=ticker)
                    logger.debug(f"Key ratios: ROE={info['data'][0].return_on_equity}, D/E={info['data'][0].debt_to_equity}",
                        module="fetch_financial_metrics_async", ticker=ticker)

                    previewed = True
                    break
            if previewed:
                break

    # Convert results to dictionary
    return {ticker: metrics for ticker, metrics in results} # Review: What does this line of code do?

# ===== GET OUTSTANDING SHARES =====
def get_outstanding_shares(
    ticker: str, 
    date: str = None,   
    verbose_data: bool = False,
    balance_sheet=None,
    balance_sheet_column=None
    ) -> Optional[float]:
    """
    Get historical outstanding shares count for a company. 
    Companies already adjust their balance sheets to stock splits.
    
    Args:
        ticker: Stock ticker symbol
        date: Date to get shares for (YYYY-MM-DD), defaults to current date
        verbose_data: Whether to print detailed data information
        
    Returns:
        Number of outstanding shares or None if not available
    """
        
    logger.debug(f"Getting outstanding shares for {ticker} at {date}", 
                module="get_outstanding_shares", ticker=ticker)

    # If balance sheet data is provided, use it directly (for fetch_financial_metrics_async)
    if balance_sheet is not None and balance_sheet_column is not None:
        logger.debug(f"Using provided balance sheet data for {ticker} at {date}", 
                    module="get_outstanding_shares", ticker=ticker)
        
        # Get shares from the provided balance sheet column
        shares = None
        share_field_names = [
            'Ordinary Shares Number',        # Primary field - most common
            'Share Issued',                  # First fallback - total shares issued
        ]
        
        for field in share_field_names:
            if field in balance_sheet.index:
                shares = balance_sheet[balance_sheet_column].get(field)
                if shares and shares > 0:
                    logger.debug(f"Found shares ({shares:,.0f}) using field '{field}' from provided balance sheet",
                               module="get_outstanding_shares", ticker=ticker)
                    # Cache the result
                    _cache.set_outstanding_shares(ticker, date, float(shares))
                    return float(shares)
        
        # Smart logging based on data age - older data missing is expected
        try:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            years_ago = (datetime.now() - date_obj).days / 365.25
            
            if years_ago > 4:  # More than 4 years old
                logger.debug(f"No valid shares data in provided balance sheet for {ticker} at {date} (expected for data >4 years old)",
                            module="get_outstanding_shares", ticker=ticker)
            else:  # Recent data missing is unexpected
                logger.warning(f"No valid shares data in provided balance sheet for {ticker} at {date}",
                              module="get_outstanding_shares", ticker=ticker)
        except ValueError:
            logger.warning(f"No valid shares data in provided balance sheet for {ticker} at {date}",
                          module="get_outstanding_shares", ticker=ticker)
        return None

    # Fallback to original logic for other callers
    cached_shares = _cache.get_outstanding_shares(ticker, date)
    if cached_shares:
        logger.debug(f"Retrieved outstanding shares from cache: {cached_shares:}", 
                    module="get_outstanding_shares", ticker=ticker)
        return cached_shares
    else:
        logger.debug(f"Computing outstanding shares", module="get_outstanding_shares", ticker=ticker)

    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Get annual balance sheets
        annual_balance_sheet = ticker_obj.balance_sheet
        if annual_balance_sheet.empty:
            logger.warning(f"No annual data available", 
                          module="get_outstanding_shares", ticker=ticker)
            return None
        else:
            if verbose_data:
                logger.debug(f"Available fields in balance sheet: {annual_balance_sheet.index.tolist()}", module="get_outstanding_shares", ticker=ticker)
                logger.debug(f"We got data: {annual_balance_sheet}", module="get_outstanding_shares", ticker=ticker)
        

        # Convert all annual dates to strings for easier comparison
        annual_dates = [(a_date, a_date.strftime('%Y-%m-%d')) for a_date in annual_balance_sheet.columns]

        # Sort by date string (newer dates first)
        annual_dates.sort(key=lambda x: x[1], reverse=True)

        # Find the most recent annual report before or equal to the specified date
        annual_date = None
        for a_date, a_date_str in annual_dates:
            if a_date_str <= date:
                annual_date = a_date
                break

        if not annual_date:
            logger.warning(f"No annual data before {date} for {ticker}", 
                          module="get_outstanding_shares", ticker=ticker)
            return None
        else:
            if verbose_data:
                logger.debug(f"Selected annual date: {annual_date} for target date: {date}",
                module="get_outstanding_shares", ticker=ticker)
            if annual_date.strftime('%Y-%m-%d') != date:
                logger.debug(f"Using {annual_date.strftime('%Y-%m-%d')} annual data for {date} (most recent available)",
                module="get_outstanding_shares", ticker=ticker)
        
        # Get shares from that annual report by checking multiple potential field names
        shares = None
        share_field_names = [
            'Ordinary Shares Number',        # Primary field - most common
            'Share Issued',                  # First fallback - total shares issued
        ]
        
        for field in share_field_names:
            if field in annual_balance_sheet.index:
                shares = annual_balance_sheet[annual_date].get(field)
                if shares and shares > 0:
                    if verbose_data:
                        logger.debug(f"Found shares ({shares:,.0f}) using field '{field}' for annual report ending {annual_date.strftime('%Y-%m-%d')}",
                                   module="get_outstanding_shares", ticker=ticker)
                    break
        
        if not shares or shares <= 0:
            logger.warning(f"No valid shares data in annual report ending {annual_date.strftime('%Y-%m-%d')} for {ticker}",
                          module="get_outstanding_shares", ticker=ticker)
            return None
        
        # Cache the result
        _cache.set_outstanding_shares(ticker, date, float(shares))
            
        logger.debug(f"Final outstanding shares for {ticker} at {date}: {shares}", 
                    module="get_outstanding_shares", ticker=ticker)
        
        if verbose_data:
            logger.debug("=== OUTSTANDING SHARES DETAILS ===", module="get_outstanding_shares", ticker=ticker)
            logger.debug(f"Date requested: {date}", module="get_outstanding_shares", ticker=ticker)
            logger.debug(f"Annual date used: {annual_date.strftime('%Y-%m-%d')}", module="get_outstanding_shares", ticker=ticker)
            
        return float(shares)
    
    except Exception as e:
        logger.error(f"Error calculating outstanding shares for {ticker} at {date}: {e}", 
                    module="get_outstanding_shares", ticker=ticker)
        return None


# ===== MARKET CAP FUNTIONS =====
def get_market_cap(
    ticker: str, 
    end_date: str,
    verbose_data: bool = False
    ) -> float:
    """
    Fetch market cap for a ticker at a specific date.
    yfinance only provides the market cap for the current day.
    For any other date, get_historical_market_cap is called.
    """

    logger.debug(f"Running get_market_cap() for {ticker} at {end_date}", 
                module="get_market_cap", ticker=ticker)

    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Try using info first (current market cap, not historical)
        market_cap = ticker_obj.info.get('marketCap')
        logger.debug(f"Market cap for {[ticker]} is {market_cap}", module="get_market_cap")
        
        # If we need a historical market cap, calculate it
        if market_cap is None or end_date != datetime.now().strftime('%Y-%m-%d'):
            market_cap = get_historical_market_cap(ticker, end_date, verbose_data)
            logger.debug(f"Market cap for {ticker} is either None or the end_date isn't the current date.",
                         module="get_market_cap")
        
        return market_cap
    except Exception as e:
        logger.error(f"Error fetching market cap for {ticker}: {e}", 
                    module="get_market_cap", ticker=ticker)
        return None


def get_historical_market_cap(ticker: str, date: str, verbose_data: bool = False, balance_sheet=None, balance_sheet_column=None) -> Optional[float]:
    """
    Get historical market cap for a ticker at a specific date by retrieving shares outstanding 
    from the most recent quarterly balance sheet and adjusting for any stock splits that 
    occurred between that quarter's end date and the requested date.
    """

    logger.debug(f"Running get_historical_market_cap() for {ticker} at {date}", 
                module="get_historical_market_cap", ticker=ticker)

    # overwritten to False for the sake of clarity
    verbose_data = False

    try:
        # Check if the requested date is a trading day
        if not is_trading_day(date):
            # Get the previous trading day
            adjusted_date = get_previous_trading_day(date)
            logger.debug(f"Date {date} is not a trading day, using previous trading day: {adjusted_date}", 
                        module="get_historical_market_cap", ticker=ticker)
            date = adjusted_date
        
        end_date_dt = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1) # Since end_date in yfinance is exclusive
        end_date_str = end_date_dt.strftime('%Y-%m-%d')

        # If the end date is also not a trading day, adjust it to next trading day
        # Note: This does not create lookahead bias because we only use data for 'date' (.iloc[0]),
        # not for 'end_date_str'. The end_date_str is only needed for yfinance API requirements.
        if not is_trading_day(end_date_str):
            end_date_str = get_next_trading_day(date)
            logger.debug(f"Adjusted end date to next trading day: {end_date_str}", 
                        module="get_historical_market_cap", ticker=ticker)

        ticker_obj = yf.Ticker(ticker)

        # Get outstanding shares
        shares = get_outstanding_shares(ticker, date, verbose_data, balance_sheet, balance_sheet_column)
        
        if not shares:
            # Smart logging based on data age - older data missing is expected
            try:
                date_obj = datetime.strptime(date, '%Y-%m-%d')
                years_ago = (datetime.now() - date_obj).days / 365.25
                
                if years_ago > 4:  # More than 4 years old
                    logger.debug(f"Could not determine shares outstanding for {ticker} at {date} (expected for data >4 years old)", 
                                module="get_historical_market_cap", ticker=ticker)
                else:  # Recent data missing is unexpected
                    logger.warning(f"Could not determine shares outstanding for {ticker} at {date}", 
                                  module="get_historical_market_cap", ticker=ticker)
            except ValueError:
                logger.warning(f"Could not determine shares outstanding for {ticker} at {date}", 
                              module="get_historical_market_cap", ticker=ticker)
            return None
            

        # Get historical price
        hist = ticker_obj.history(start=date, end=end_date_str)
        
        if hist.empty:
            logger.warning(f"${ticker}: possibly delisted; no price data found  (1d {date} -> {end_date_str})", 
                          module="get_historical_market_cap", ticker=ticker)
            return None
        
        close_price = hist['Close'].iloc[0]
        market_cap = close_price * shares

        # Add detailed logging for verbose_data mode
        if verbose_data:
            logger.debug("=== MARKET CAP CALCULATION DETAILS ===", module="get_historical_market_cap", ticker=ticker)
            logger.debug(f"Date requested: {date}", module="get_historical_market_cap", ticker=ticker)
            logger.debug(f"Closing price: {close_price}", module="get_historical_market_cap", ticker=ticker)
            logger.debug(f"Shares outstanding: {shares}", module="get_historical_market_cap", ticker=ticker)
            logger.debug(f"Share price on {date}: ${close_price}", module="get_historical_market_cap", ticker=ticker)
            logger.debug(f"Calculated market cap: ${market_cap}", module="get_historical_market_cap", ticker=ticker)
            

        return market_cap
    
    except Exception as e:
        logger.error(f"Error calculating historical market cap for {ticker} at {date}: {e}", 
                    module="get_historical_market_cap", ticker=ticker)
        return None


# ===== INSIDER TRADES FROM SEC EDGAR =====
def get_insider_trades(
    ticker: Union[str, List[str]],
    end_date: str,
    start_date: str = None,
    limit: int = 1000,
    verbose_data: bool = False,
    ) -> List[InsiderTrade]:
    """
    Fetch insider trades for a single ticker.
    Delegates caching to fetch_multiple_insider_trades.
    
    Args:
        ticker: Stock ticker symbol
        end_date: End date string (YYYY-MM-DD)
        start_date: Optional start date (YYYY-MM-DD)
        limit: Maximum number of trades to return
        verbose_data: Optional flag for verbose logging
        
    Returns:
        List of InsiderTrade objects
    """
    logger.debug(f"Running get_insider_trades() to {end_date}", 
                module="get_insider_trades", ticker=ticker)
    
    try:
        # Use the SEC EDGAR scraper to fetch insider trades (handles caching internally)
        edgar_results = fetch_multiple_insider_trades(ticker, end_date, start_date, verbose_data)
        trades = edgar_results.get(ticker, []) or []
        
        logger.debug(f"get_insider_trades received results: {len(trades)} trades", 
                    module="get_insider_trades", ticker=ticker)
        
        return trades[:limit]
    
    except Exception as e:
        logger.error(f"Error fetching insider trades for {ticker}: {e}", 
                    module="get_insider_trades", ticker=ticker)
        return []


# ===== COMPANY NEWS =====
def get_company_news(
    tickers: Union[str, List[str]],
    end_date: str,
    start_date: str = None,
    limit: int = 1000,
    verbose_data: bool = False
    ) -> Union[List[CompanyNews], Dict[str, List[CompanyNews]]]:
    """
    Fetch company news sentiment from database or Alpha Vantage API.
    Unified function that handles both single ticker and multiple tickers.
    
    Args:
        tickers: Single ticker string or list of ticker symbols
        end_date: End date string (YYYY-MM-DD) 
        start_date: Optional start date (YYYY-MM-DD)
        limit: Maximum number of news items to return per ticker
        verbose_data: Optional flag for verbose logging
        
    Returns:
        If single ticker: List of CompanyNews objects
        If multiple tickers: Dictionary mapping tickers to lists of CompanyNews objects
    """
    
    # Handle single ticker vs multiple tickers
    is_single_ticker = isinstance(tickers, str)
    ticker_list = [tickers] if is_single_ticker else tickers
    
    logger.debug(f"Running get_company_news() from {start_date} to {end_date} for {len(ticker_list)} tickers", 
                module="get_company_news")

    # Get Alpha Vantage API key
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.error("No Alpha Vantage API key found in environment variable ALPHA_VANTAGE_API_KEY", 
                    module="get_company_news")
        empty_result = [] if is_single_ticker else {ticker: [] for ticker in ticker_list}
        return empty_result

    # Initialize variables
    results = {}
    cache_hits = 0
    total_tickers = len(ticker_list)
    
    # Check database cache first
    for ticker in ticker_list:
        if is_news_cache_valid(ticker, max_age_days=1):
            cached_news = get_company_news_from_db(ticker)
            
            # Filter by date range
            filtered_data = []
            for news in cached_news:
                try:
                    # Check if news is within date range
                    if (start_date is None or news.date >= start_date) and news.date <= end_date:
                        filtered_data.append(news)
                except Exception as e:
                    logger.warning(f"Error processing cached news item for {ticker}: {e}", 
                                  module="get_company_news", ticker=ticker)
                    continue
            
            # Always add to results if cache is valid (even if empty)
            results[ticker] = filtered_data[:limit]
            cache_hits += 1
            
            if filtered_data:
                logger.debug(f"Using {len(filtered_data)} cached news items from database for {ticker}", 
                           module="get_company_news", ticker=ticker)
            else:
                logger.debug(f"No news available for {ticker}", module="get_company_news", ticker=ticker)
            continue
            
    # Fetch from Alpha Vantage for remaining tickers
    tickers_to_fetch = [ticker for ticker in ticker_list if ticker not in results]
    
    # Log cache statistics
    if verbose_data:
        cache_percentage = (cache_hits / total_tickers * 100) if total_tickers > 0 else 0
        download_percentage = 100 - cache_percentage
        logger.debug(f"COMPANY NEWS CACHE STATS:", module="get_company_news")
        logger.debug(f"  - From cache: {cache_hits}/{total_tickers} tickers ({cache_percentage:.1f}%)", module="get_company_news")   
        logger.debug(f"  - To download: {len(tickers_to_fetch)}/{total_tickers} tickers ({download_percentage:.1f}%)", module="get_company_news")
    
    # Fetch data with rate limiting
    for i, ticker in enumerate(tickers_to_fetch):
        try:
            # Add delay between requests to respect rate limits
            if i > 0:
                time.sleep(1)  # Conservative rate limiting
            
            # Alpha Vantage uses format YYYYMMDDTHHMM
            if start_date:
                time_from = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%dT0000')
            else:
                # Default to 30 days before end_date if start_date is None
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                start_dt = end_dt - timedelta(days=30)
                time_from = start_dt.strftime('%Y%m%dT0000')

            time_to = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%dT2359')
            
            # Build Alpha Vantage URL  
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                "time_from": time_from,  
                "time_to": time_to,   
                'apikey': api_key,
                'sort': 'LATEST',
                'limit': limit
            }
            
            logger.debug(f"Fetching news sentiment from Alpha Vantage", 
                        module="get_company_news", ticker=ticker)
            
            # Make API request
            logger.debug(f"Making API request for {ticker} with params: {params}", 
                        module="get_company_news", ticker=ticker)
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Alpha Vantage HTTP error for {ticker}: {response.status_code} - {response.text[:200]}", 
                           module="get_company_news", ticker=ticker)
                results[ticker] = []
                continue
                
            try:
                data = response.json()
            except Exception as e:
                logger.error(f"Failed to parse JSON response for {ticker}: {e}", 
                           module="get_company_news", ticker=ticker)
                logger.error(f"Raw response: {response.text[:500]}", 
                           module="get_company_news", ticker=ticker)
                results[ticker] = []
                continue

            # Debug: Log the raw API response structure
            logger.debug(f"API response keys for {ticker}: {list(data.keys())}", 
                        module="get_company_news", ticker=ticker)
            
            feed_count = len(data.get('feed', []))
            logger.debug(f"Alpha Vantage returned {feed_count} raw news items for {ticker}", 
                        module="get_company_news", ticker=ticker)
            
            if verbose_data and feed_count > 0:
                sample_item = data['feed'][0]
                logger.debug(f"Sample raw item: title='{sample_item.get('title', 'N/A')}', date='{sample_item.get('time_published', 'N/A')}'", 
                             module="get_company_news", ticker=ticker)
            
            # Check for API errors
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage API error for {ticker}: {data['Error Message']}", 
                           module="get_company_news", ticker=ticker)
                results[ticker] = []
                continue
                
            if 'Information' in data:
                logger.warning(f"Alpha Vantage API response: {data}", 
                             module="get_company_news", ticker=ticker)
                results[ticker] = []
                continue
                
            # Check if feed is empty (legitimate no news vs API issue)
            if feed_count == 0:
                logger.debug(f"No news articles found for {ticker} in date range {start_date} to {end_date}", 
                            module="get_company_news", ticker=ticker)
                logger.debug(f"This could be due to: 1) No news published, 2) Date range outside API limits, 3) Ticker not covered", 
                            module="get_company_news", ticker=ticker)

            # Process news items
            news_list = []
            feed_items = data.get('feed', [])
            
            for item in feed_items:
                try:
                    # Parse the timestamp  
                    time_published = item.get('time_published', '')
                    if len(time_published) >= 8:
                        # Format: YYYYMMDDTHHMMSS
                        date_str = time_published[:8]  # Get YYYYMMDD part
                        date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    else:
                        continue
                    
                    # Get ticker-specific sentiment
                    ticker_sentiments = item.get('ticker_sentiment', [])
                    ticker_sentiment = None
                    
                    for ts in ticker_sentiments:
                        if ts.get('ticker') == ticker:
                            sentiment_label = ts.get('ticker_sentiment_label', 'Neutral')
                            # Map Alpha Vantage labels to your system
                            if sentiment_label in ['Bullish', 'Somewhat-Bullish']:
                                ticker_sentiment = 'positive'
                            elif sentiment_label in ['Bearish', 'Somewhat-Bearish']:
                                ticker_sentiment = 'negative'  
                            else:
                                ticker_sentiment = 'neutral'
                            break
                    
                    # Use overall sentiment if ticker-specific not found
                    if ticker_sentiment is None:
                        overall_sentiment = item.get('overall_sentiment_label', 'Neutral')
                        if overall_sentiment in ['Bullish', 'Somewhat-Bullish']:
                            ticker_sentiment = 'positive'
                        elif overall_sentiment in ['Bearish', 'Somewhat-Bearish']:
                            ticker_sentiment = 'negative'
                        else:
                            ticker_sentiment = 'neutral'

                    news = CompanyNews(
                        ticker=ticker,
                        title=item.get('title', ''),
                        author=item.get('authors', [''])[0] if item.get('authors') else '',
                        source=item.get('source', ''),
                        date=date,
                        url=item.get('url', ''),
                        sentiment=ticker_sentiment
                    )
                    news_list.append(news)

                except Exception as e:
                    logger.warning(f"Error processing news item for {ticker}: {e}", 
                                    module="get_company_news", ticker=ticker)
                    continue
                
            # Save to database
            saved_count = save_company_news(ticker, news_list)
            logger.debug(f"Saved {saved_count} new news items to database for {ticker} (total fetched: {len(news_list)})", 
                        module="get_company_news", ticker=ticker)
                
            # Add to results
            results[ticker] = news_list
                
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}", 
                module="get_company_news", ticker=ticker)
            logger.error(f"Traceback: {traceback.format_exc()}", 
                        module="get_company_news", ticker=ticker)
            results[ticker] = []
    
    # Return appropriate format based on input
    if is_single_ticker:
        return results.get(tickers, [])
    else:
        return results


# ===== USED BY AGENTS =====
def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    verbose_data: bool = False,
    ) -> List[LineItem]:
    """
    Fetch specific line items from financial statements using caching and yfinance.
    
    Args:
        ticker: Stock ticker symbol
        line_items: List of line items to retrieve
        end_date: End date string (YYYY-MM-DD)
        period: Period type ("ttm", "annual", etc.)
        limit: Maximum number of periods to return
        
    Returns:
        List of LineItem objects
    """

    # overwritten to False for the sake of clarity
    verbose_data = False

    logger.debug(f"Running search_line_items() for {ticker}: {line_items}", 
                module="search_line_items", ticker=ticker)
    
    # First check if we have this data in cache
    cached_line_items = _cache.get_line_items(ticker, end_date, period, line_items)
    if cached_line_items:
        logger.debug(f"Retrieved line items for {ticker} from cache", 
                  module="search_line_items", ticker=ticker)

        if cached_line_items and verbose_data:
            logger.debug(f"Cash flow statement structure from cache:", module="search_line_items", ticker=ticker)
            # Check if any cached item has dividends_and_other_cash_distributions
            has_dividends = any('dividends_and_other_cash_distributions' in item for item in cached_line_items)
            logger.debug(f"Cached line items have dividends attribute: {has_dividends}", module="search_line_items", ticker=ticker)
            
            # Print the first cached item as an example
            if cached_line_items:
                logger.debug(f"First cached line item fields: {list(cached_line_items[0].keys())}", module="search_line_items", ticker=ticker)
                logger.debug(f"Cached line items data: {cached_line_items}", module="search_line_items", ticker=ticker)
        
        
        # Convert dictionary data to LineItem objects
        return [LineItem(**item) for item in cached_line_items]

    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Get the financial statements
        is_quarterly = period.lower() in ["q", "quarterly", "ttm"]
        
        if is_quarterly:
            income_stmt = ticker_obj.quarterly_income_stmt
            balance_sheet = ticker_obj.quarterly_balance_sheet
            cash_flow = ticker_obj.quarterly_cashflow
        else:
            income_stmt = ticker_obj.income_stmt
            balance_sheet = ticker_obj.balance_sheet
            cash_flow = ticker_obj.cashflow
        
        results = []

        # Check if we have data from yfinance
        if income_stmt.empty and balance_sheet.empty and cash_flow.empty:
            logger.warning(f"No financial data found for {ticker}", 
                          module="search_line_items", ticker=ticker)
            
        if verbose_data:
            logger.debug(f"Financial statement shapes:", module="search_line_items", ticker=ticker)
            logger.debug(f"  - Income Statement: {income_stmt.shape if not income_stmt.empty else 'Empty'}", 
                        module="search_line_items", ticker=ticker)
            logger.debug(f"  - Balance Sheet: {balance_sheet.shape if not balance_sheet.empty else 'Empty'}", 
                        module="search_line_items", ticker=ticker)
            logger.debug(f"  - Cash Flow: {cash_flow.shape if not cash_flow.empty else 'Empty'}", 
                        module="search_line_items", ticker=ticker)

            logger.debug(f"Income stmt: {income_stmt}", module="search_line_items", ticker=ticker)
            logger.debug(f"balance sheet: {balance_sheet}", module="search_line_items", ticker=ticker)
            logger.debug(f"Cash flow: {cash_flow}", module="search_line_items", ticker=ticker)

        # Get columns (reporting dates) that exist in all financial statements
        income_cols = set(income_stmt.columns) if not income_stmt.empty else set()
        balance_cols = set(balance_sheet.columns) if not balance_sheet.empty else set()
        cashflow_cols = set(cash_flow.columns) if not cash_flow.empty else set()
        
        # Find the intersection of columns that exist in all statements that have data
        common_columns = set()
        if income_cols and balance_cols and cashflow_cols:
            common_columns = income_cols.intersection(balance_cols).intersection(cashflow_cols)
        else:
            logger.warning("No intersection of columns possible", module="search_line_items", ticker=ticker)

        # Sort columns in descending order such that the newest is first
        sorted_columns = sorted(common_columns, reverse=True)

        # Filter columns by end_date 
        valid_columns = [col for col in sorted_columns if col.strftime('%Y-%m-%d') <= end_date]

        # Filter out columns with too many NaN values
        filtered_valid_columns = []
        for col in valid_columns:
            # Check what percentage of values are NaN in each statement
            income_nan_count = income_stmt[col].isna().sum() / len(income_stmt)
            balance_nan_count = balance_sheet[col].isna().sum() / len(balance_sheet)
            cashflow_nan_count = cash_flow[col].isna().sum() / len(cash_flow)
            
            # Only include columns where at least 70% of values are not NaN
            if income_nan_count < 0.3 and balance_nan_count < 0.3 and cashflow_nan_count < 0.3:
                filtered_valid_columns.append(col)
            else:
                logger.debug(f"Skipping date {col.strftime('%Y-%m-%d')} due to excessive NaN values (I:{income_nan_count:.2f}, B:{balance_nan_count:.2f}, C:{cashflow_nan_count:.2f})",
                            module="search_line_items", ticker=ticker)

        # Limit the number of columns
        valid_columns = filtered_valid_columns[:limit]  

        if verbose_data:
            logger.debug(f"Valid columns: {[col.strftime('%Y-%m-%d') for col in valid_columns]}", 
                         module="search_line_items", ticker=ticker)
        
        # Process each valid column (date)
        for col in valid_columns:
            report_date = col.strftime('%Y-%m-%d')
                
            # Create a line item for this date
            line_item = LineItem(
                ticker=ticker,
                report_period=report_date,
                period=period,
                currency="USD"
            )

            if verbose_data:
                logger.debug(f"report date: {report_date}, end_date: {end_date}", module="search_line_items", ticker=ticker)
                logger.debug(f"OUTER LOOP: Created line_item on {report_date}", 
                       module="search_line_items", ticker=ticker)
            
            # Add requested line items
            for item in line_items:

                if verbose_data:
                    logger.debug(f"  INNER LOOP: Processing item '{item}' for {ticker} on {report_date}", 
                                 module="search_line_items", ticker=ticker)
                
                # Map line_items to yfinance fields
                item_mapping = {
                    "revenue": ("Total Revenue", income_stmt),
                    "net_income": ("Net Income", income_stmt),
                    "capital_expenditure": ("Capital Expenditure", cash_flow),
                    "depreciation_and_amortization": ([
                        "Depreciation And Amortization",
                        "Depreciation Amortization Depletion",
                        "Reconciled Depreciation"
                    ], cash_flow),
                    "outstanding_shares":  ("Ordinary Shares Number", balance_sheet),
                    "total_assets": ("Total Assets", balance_sheet),
                    "total_liabilities": ("Total Liabilities Net Minority Interest", balance_sheet),
                    "working_capital": None,  # Special calculation 
                    "debt_to_equity": None,  # Special calculation
                    "free_cash_flow": None,  # Special calculation
                    "operating_margin": None,  # Special calculation
                    "dividends_and_other_cash_distributions": None # Special calculation,
                }

                        
                if item == "dividends_and_other_cash_distributions":
                    # Try multiple possible field names
                    dividend_field_names = ["Cash Dividends Paid", "Common Stock Dividend Paid", "Dividends Paid"]
                    dividend_found = False
                    for field_name in dividend_field_names:
                        if field_name in cash_flow.index:
                            value = cash_flow.loc[field_name, col]
                            setattr(line_item, item, float(value))
                            dividend_found = True
                            break
                    
                    if not dividend_found:
                        logger.debug(f"No dividend data found for {report_date}", module="search_line_items", ticker=ticker)
                
                elif item == "working_capital":
                    # Working capital = Current Assets - Current Liabilities
                    current_assets = balance_sheet.loc["Current Assets", col] if "Current Assets" in balance_sheet.index else None
                    current_liabilities = balance_sheet.loc["Current Liabilities", col] if "Current Liabilities" in balance_sheet.index else None
                    if current_assets is not None and current_liabilities is not None:
                        setattr(line_item, item, float(current_assets - current_liabilities))
                    else:
                        logger.debug(f"Missing data for working_capital calculation at {report_date}: current_assets={current_assets}, current_liabilities={current_liabilities}", 
                        module="search_line_items", ticker=ticker)
                        # Set to None when calculation fails to ensure attribute exists
                        setattr(line_item, item, None)
                
                elif item == "debt_to_equity":
                    # Debt to Equity
                    total_liabilities = balance_sheet.loc["Total Liabilities Net Minority Interest", col] if "Total Liabilities Net Minority Interest" in balance_sheet.index else None
                    stockholders_equity = balance_sheet.loc["Stockholders Equity", col] if "Stockholders Equity" in balance_sheet.index else None
                    if total_liabilities is not None and stockholders_equity is not None and float(stockholders_equity) != 0:
                        setattr(line_item, item, float(total_liabilities) / float(stockholders_equity))
                    else:
                        logger.debug(f"Missing data for debt_to_equity calculation at {report_date}: total_liabilities={total_liabilities}, stockholders_equity={stockholders_equity}", 
                        module="search_line_items", ticker=ticker)
                
                elif item == "free_cash_flow":
                    # First try to get Free Cash Flow directly from the cash flow statement
                    if "Free Cash Flow" in cash_flow.index:
                        setattr(line_item, item, float(cash_flow.loc["Free Cash Flow", col]))
                    else:
                        # Fallback: Calculate Free Cash Flow = Operating Cash Flow + Capital Expenditure
                        operating_cash_flow = cash_flow.loc["Operating Cash Flow", col] if "Operating Cash Flow" in cash_flow.index else None
                        capital_expenditure = cash_flow.loc["Capital Expenditure", col] if "Capital Expenditure" in cash_flow.index else None
                        if operating_cash_flow is not None and capital_expenditure is not None:
                            setattr(line_item, item, float(operating_cash_flow + capital_expenditure))
                        else:
                            logger.debug(f"Missing data for free_cash_flow calculation at {report_date}: operating_cash_flow={operating_cash_flow}, capital_expenditure={capital_expenditure}", 
                            module="search_line_items", ticker=ticker)
                
                elif item == "operating_margin":
                    # Operating Margin = Operating Income / Revenue
                    operating_income = income_stmt.loc["Operating Income", col] if "Operating Income" in income_stmt.index else None
                    total_revenue = income_stmt.loc["Total Revenue", col] if "Total Revenue" in income_stmt.index else None
                    if operating_income is not None and total_revenue is not None and float(total_revenue) > 0:
                        setattr(line_item, item, float(operating_income) / float(total_revenue))
                    else:
                        logger.debug(f"Missing data for operating_margin calculation at {report_date}: operating_income={operating_income}, total_revenue={total_revenue}", 
                        module="search_line_items", ticker=ticker)
                
                elif item_mapping.get(item):
                    field_data, df = item_mapping[item]
                    # Handle both single field names and lists of field names
                    if isinstance(field_data, list):
                        # Try each field name in the list until one is found
                        for field in field_data:
                            if field in df.index:
                                value = df.loc[field, col]
                                setattr(line_item, item, float(value))
                                break
                    else:
                        # Single field name (backwards compatibility)
                        field = field_data
                        if field in df.index:
                            value = df.loc[field, col]
                            setattr(line_item, item, float(value))
            
            results.append(line_item)

            if verbose_data:
                logger.debug(f"line_item: {line_item}", module="search_line_items", ticker=ticker)

        logger.debug(f"Retrieved {len(results)} line items for {ticker}", 
                    module="search_line_items", ticker=ticker)

        if verbose_data or len(results) == 0:
            logger.debug(f"Found {len(results)} items", module="search_line_items", ticker=ticker)
            logger.debug(f"Results {results}", module="search_line_items", ticker=ticker)

        # Cache the downloaded data
        if results:
            cache_data = [item.model_dump() for item in results]
            _cache.set_line_items(ticker, end_date, period, line_items, cache_data)
        
        return results
    
    except Exception as e:
        logger.error(f"Error searching line items for {ticker}: {e}", 
                module="search_line_items", ticker=ticker)
        return []


# ===== REVIEW: LIKELY A USELESS FUNCTION =====
async def get_price_data_batch(tickers: List[str], start_date: str, end_date: str, verbose_data: bool = False) -> Dict[str, pd.DataFrame]:
    """Get price data as DataFrames for multiple tickers in one batch request"""

    from utils.logger import logger
    logger.debug(f"Running get_price_data_batch() for {len(tickers)} tickers from {start_date} to {end_date}", 
                module="get_price_data_batch")
    return await fetch_prices_batch(tickers, start_date, end_date, verbose_data)

# ===== DATA FETCHING FUNCTION FOR SRC/BACKTESTER.PY =====
def get_data_for_tickers(tickers: List[str], start_date: str, end_date: str, batch_size: int = 20, verbose_data: bool = False):
    """
    Process multiple tickers efficiently with batching and parallel processing.
    Returns a dictionary with all data for each ticker.
    Is used by the prefetch_data() function in src/backtester to fetch all data needed to run a backtest.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        batch_size: Number of tickers to process in each batch
        verbose_data: Optional flag for verbose logging
        
    Returns:
        Dictionary of ticker to data mapping
    """
    
    logger.debug(f"Running get_data_for_tickers() for {len(tickers)} tickers from {start_date} to {end_date}", 
                module="get_data_for_tickers")

    # Split tickers into batches of batch_size
    ticker_batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    
    results = {}
    try:
        # Process each batch of tickers
        for batch in ticker_batches:
            # Get insider trades (this is synchronous)
            insider_results = fetch_multiple_insider_trades(batch, end_date, start_date, verbose_data)
            
            # Create and run a new event loop for the async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run price data, financial metrics, and news tasks synchronously
                price_data = loop.run_until_complete(fetch_prices_batch(batch, start_date, end_date, verbose_data))
                metrics_data = loop.run_until_complete(fetch_financial_metrics_async(batch, end_date, "annual", 10, verbose_data))
                news_data = get_company_news(batch, end_date, start_date, 1000, verbose_data)
            finally:
                loop.close()
            
            # Merge into results dictionary
            for ticker in batch:
                results[ticker] = {
                    "prices": price_data.get(ticker, pd.DataFrame()),
                    "metrics": metrics_data.get(ticker, []),
                    "insider_trades": insider_results.get(ticker, []),
                    "news": news_data.get(ticker, [])
                }
    except Exception as e:
        logger.error(f"Error fetching data for tickers: {e}", module="get_data_for_tickers")
        logger.error(f"Traceback: {traceback.format_exc()}", module="get_data_for_tickers")

    # Preview the results
    if results and verbose_data:
        example_ticker = next(iter(results))
        logger.debug("--- DATA PREVIEW from get_data_for_tickers ---", module="get_data_for_tickers")
        logger.debug(f"Sample data for ticker: {example_ticker}", 
                    module="get_data_for_tickers", ticker=example_ticker)
        
        # Preview prices
        if isinstance(results[example_ticker]["prices"], pd.DataFrame) and not results[example_ticker]["prices"].empty:
            prices_df = results[example_ticker]["prices"]
            logger.debug(f"Prices sample:",
                module="get_data_for_tickers", ticker=example_ticker)
            logger.debug(f"Prices data shape: {prices_df.shape}",
                module="get_data_for_tickers", ticker=example_ticker)
            logger.debug(f"First row: {prices_df.head(1)}",
                module="get_data_for_tickers", ticker=example_ticker)
            
        # Preview metrics
        metrics = results[example_ticker]["metrics"]
        if metrics:
            metrics_dump = metrics[0].model_dump() if hasattr(metrics[0], 'model_dump') else metrics[0]
            logger.debug(
                f"Metrics sample:\n {metrics_dump}",
                module="get_data_for_tickers", 
                ticker=example_ticker
            )
            
        # Preview insider trades
        insider_trades = results[example_ticker]["insider_trades"]
        if insider_trades:
            logger.debug(
                f"Insider trades count: {len(insider_trades)}",
                module="get_data_for_tickers", 
                ticker=example_ticker
            )
            if insider_trades:
                trades_dump = insider_trades[0].model_dump() if hasattr(insider_trades[0], 'model_dump') else insider_trades[0]
                logger.debug(
                    f"Sample trade: {trades_dump}",
                    module="get_data_for_tickers", 
                    ticker=example_ticker
                )
        
        # Preview news
        news = results[example_ticker]["news"]
        if news:
            logger.debug(
                f"News count: {len(news)}",
                module="get_data_for_tickers", 
                ticker=example_ticker
            )
            if news:
                news_dump = news[0].model_dump() if hasattr(news[0], 'model_dump') else news[0]
                logger.debug(
                    f"Sample news: {news_dump}",
                    module="get_data_for_tickers", 
                    ticker=example_ticker
                )
    
    return results

