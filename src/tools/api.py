import os
import pandas as pd
import yfinance as yf
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Dict, Optional, Any, Union
import datetime
from pathlib import Path

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

# Import the market calendar, SEC EDGAR scraper and logger.
from utils.market_calendar import is_trading_day, adjust_date_range
from tools.sec_edgar_scraper import fetch_insider_trades_for_ticker, fetch_multiple_insider_trades
from utils.logger import logger


# ===== CACHE INITIALIZATION =====
# Create a global cache instance with disk persistence
_cache = DiskCache(cache_dir=Path("./cache"))


# ===== PRICE FUNCTIONS =====
def get_prices(ticker: str, start_date: str, end_date: str, verbose_data: bool = False) -> List[Price]:
    """
    Fetch price data for a single ticker.
    This is the main public API that maintains compatibility with the original function.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        List of Price objects
    """

    # Log function call
    logger.debug(f"Fetching prices for {ticker} from {start_date} to {end_date}", module="get_prices", ticker=ticker)

    # Run the async function in a new event loop
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(get_prices_async([ticker], start_date, end_date, verbose_data))
        return results.get(ticker, []) # Review: why is it not just return results? Why .get()?
    finally:
        loop.close() # Review: Why do I have to close the loop? Why can't I just omit this finally:?


async def get_prices_async(tickers: List[str], start_date: str, end_date: str, verbose_data: bool = False) -> Dict[str, List[Price]]:
    """Fetch prices for multiple tickers asynchronously"""

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
            logger.debug(f"Either stard_date {start_date} or end_date {end_date} isn't a trading day.")
            start_date, end_date = adjust_date_range(start_date, end_date)
        except ValueError as e:
            logger.warning(f"Warning: {e} - continuing with original dates", module="get_prices_async")

    df_results = await fetch_prices_batch(tickers, start_date, end_date, verbose_data) # Review: Why do I need "await" here?
    return {ticker: df_to_price_objects(df) for ticker, df in df_results.items()}


async def fetch_prices_batch(tickers: List[str], start_date: str, end_date: str, verbose_data: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical prices for multiple tickers in a single batch request.
    Uses yfinance's batch download capability for efficiency.
    Adjusts dates to trading days to handle market holidays.
    """

    logger.debug(f"Running fetch_prices_batch() for {len(tickers)} tickers", module="fetch_prices_batch")

    # First check cache for all tickers
    cached_results = {}
    tickers_to_fetch = []
    
    # Cache tracking counters
    cache_hits = 0
    total_tickers = len(tickers)

    # Debug: cache structure
    if verbose_data:
        for ticker in tickers[:3]:  # Just check a few tickers to understand cache behavior
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
    
    # Fetch data for tickers not in cache
    try:
        # yfinance batch download
        logger.debug(f"Downloading price data for {len(tickers_to_fetch)} tickers", 
                         module="fetch_prices_batch")
        data = yf.download(
            tickers_to_fetch, 
            start=start_date, 
            end=end_date, 
            group_by='ticker',
            auto_adjust=True,
            progress=False
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

    logger.debug(f"Running get_price_data() for {ticker} from {start_date} to {end_date}", 
                module="get_price_data", ticker=ticker)

    prices = get_prices(ticker, start_date, end_date, verbose_data)
    return prices_to_df(prices)


# ===== DATA CONVERSION UTILITIES =====
def df_to_cache_format(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to a format suitable for caching"""
    df = df.reset_index()
    df['time'] = df['Date'].dt.strftime('%Y-%m-%d')
    result = df.to_dict('records')
    return result


def df_to_price_objects(df: pd.DataFrame) -> List[Price]:
    """Convert DataFrame to a list of Price objects"""
    prices = []
    for _, row in df.iterrows():
        price = Price(
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=int(row['volume']),
            time=row.name.strftime('%Y-%m-%d')
        )
        prices.append(price)
    return prices


def prices_to_df(prices: List[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame. Same as original function."""
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
    period: str = "ttm",
    limit: int = 10,
    verbose_data: bool = False
) -> List[FinancialMetrics]:
    """
    Fetch financial metrics for a single ticker.
    Maintains compatibility with the original function.
    
    Args:
        ticker: Stock ticker symbol
        end_date: End date string (YYYY-MM-DD)
        period: Period type ("ttm", "annual", etc.)
        limit: Maximum number of periods to return
        
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


async def fetch_financial_metrics_async(tickers: List[str], end_date: str, period: str = "ttm", limit: int = 10, verbose_data: bool = False):
    """Fetch financial metrics for multiple tickers in parallel"""

    logger.debug(f"Running fetch_financial_metrics_async() for {len(tickers)} tickers", 
                module="fetch_financial_metrics_async")

    # Counter for cache tracking
    cache_hits = 0
    total_tickers = len(tickers)
    results_dict = {}

    async def fetch_ticker_metrics(ticker):
        nonlocal cache_hits, results_dict # Review: What does nonlocal do? What is it for?

        cached_data = _cache.get_financial_metrics(ticker)
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
                    
                    # Calculate financial ratios
                    market_cap = get_historical_market_cap(ticker, report_date, verbose_data=verbose_data)
                    
                    # ROE, margins, etc.
                    return_on_equity = net_income / stockholders_equity if stockholders_equity and net_income else None
                    net_margin = net_income / total_revenue if total_revenue and net_income else None
                    operating_income = float(income_stmt.loc['Operating Income', col]) if 'Operating Income' in income_stmt.index else None
                    operating_margin = operating_income / total_revenue if total_revenue and operating_income else None
                    debt_to_equity = total_liabilities / stockholders_equity if stockholders_equity and total_liabilities else None
                    
                    # Growth metrics (would need previous period data for accurate calculation)
                    revenue_growth = None
                    earnings_growth = None
                    
                    # Free cash flow
                    operating_cash_flow = float(cash_flow.loc['Operating Cash Flow', col]) if 'Operating Cash Flow' in cash_flow.index else None
                    capital_expenditure = float(cash_flow.loc['Capital Expenditure', col]) if 'Capital Expenditure' in cash_flow.index else None
                    free_cash_flow = (operating_cash_flow + capital_expenditure) if operating_cash_flow is not None and capital_expenditure is not None else None
                    
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
                        current_ratio=None,  # Need current assets and liabilities
                        revenue_growth=revenue_growth,
                        earnings_growth=earnings_growth,
                        free_cash_flow_per_share=None,  # Need shares outstanding
                        earnings_per_share=None,  # Calculated based on actual reported data
                        book_value_per_share=None,  # Need shares outstanding
                        # Many fields set to None as they require additional calculation
                        **{field: None for field in [
                            'peg_ratio', 'asset_turnover', 'inventory_turnover', 'receivables_turnover',
                            'days_sales_outstanding', 'operating_cycle', 'working_capital_turnover',
                            'quick_ratio', 'cash_ratio', 'operating_cash_flow_ratio', 'debt_to_assets',
                            'interest_coverage', 'book_value_growth', 'earnings_per_share_growth', 
                            'free_cash_flow_growth', 'operating_income_growth', 'ebitda_growth', 'payout_ratio'
                        ]}
                    )
                    financial_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"Error processing financial data for {ticker} at {report_date}: {e}", 
                                module="fetch_financial_metrics_async", ticker=ticker)
                    continue
            
            # Cache the results
            if financial_metrics:
                _cache.set_financial_metrics(ticker, [metric.model_dump() for metric in financial_metrics])
            
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
    verbose_data: bool = False
) -> Optional[float]:
    """
    Get historical outstanding shares count for a company.
    
    Args:
        ticker: Stock ticker symbol
        date: Date to get shares for (YYYY-MM-DD), defaults to current date
        use_cache: Whether to use the cache
        debug_mode: Whether to print debugging information
        verbose_data: Whether to print detailed data information
        
    Returns:
        Number of outstanding shares or None if not available
    """
        
    logger.debug(f"Getting outstanding shares for {ticker} at {date}", 
                module="get_outstanding_shares", ticker=ticker)


    cached_shares = _cache.get_outstanding_shares(ticker, date)
    if cached_shares:
        logger.debug(f"Retrieved outstanding shares from cache: {cached_shares:}", 
                    module="get_outstanding_shares", ticker=ticker)
        return cached_shares

    try:
        # Get ticker object from yfinance
        ticker_obj = yf.Ticker(ticker)
        
        # Get quarterly balance sheets
        quarters = ticker_obj.quarterly_balance_sheet
        if quarters.empty:
            logger.warning(f"No quarterly data available for {ticker}", 
                          module="get_outstanding_shares", ticker=ticker)
            return None
        
        # Convert all quarter dates to strings for easier comparison
        quarter_dates = [(q_date, q_date.strftime('%Y-%m-%d')) for q_date in quarters.columns]
        # Sort by date string (newer dates first)
        quarter_dates.sort(key=lambda x: x[1], reverse=True)

        # Find the most recent quarter before or equal to the specified date
        quarter_date = None
        for q_date, q_date_str in quarter_dates:
            if q_date_str <= date:
                quarter_date = q_date
                break

        if not quarter_date:
            logger.warning(f"No quarterly data before {date} for {ticker}", 
                          module="get_outstanding_shares", ticker=ticker)
            return None
        
        # Get shares from that quarter by checking multiple potential field names
        shares = None
        share_field_names = [
            'Share Issued',
            'Shares Issued', 
            'Common Stock Shares Outstanding',
            'ShareIssued',
            'SharesIssued',
            'CommonStockSharesOutstanding',
            'Common Stock',
            'CommonStock'
        ]
        
        for field in share_field_names:
            if field in quarters.index:
                shares = quarters[quarter_date].get(field)
                if shares and shares > 0:
                    if verbose_data:
                        logger.debug(f"Found shares ({shares:,.0f}) using field '{field}' for quarter ending {quarter_date.strftime('%Y-%m-%d')}",
                                   module="get_outstanding_shares", ticker=ticker)
                    break
        
        if not shares or shares <= 0:
            logger.warning(f"No valid shares data in quarter ending {quarter_date.strftime('%Y-%m-%d')} for {ticker}",
                          module="get_outstanding_shares", ticker=ticker)
            return None
        
        # Adjust for splits between quarterly data date and specified date
        splits = ticker_obj.splits
        if not splits.empty:
            # Convert quarter_date to string for comparison
            q_date_str = quarter_date.strftime('%Y-%m-%d')
            
            # Find and apply relevant splits
            relevant_splits = []
            for split_date, ratio in zip(splits.index, splits.values):
                split_date_str = split_date.strftime('%Y-%m-%d')
                if q_date_str < split_date_str <= date:
                    relevant_splits.append((split_date, ratio))
            
            # Apply the splits to our share count
            for split_date, split_ratio in relevant_splits:
                original_shares = shares
                shares *= split_ratio
                if verbose_data:
                    logger.debug(f"Adjusted shares from {original_shares} to {shares} due to {split_ratio}:1 split on {split_date.strftime('%Y-%m-%d')}",
                            module="get_outstanding_shares", ticker=ticker)
        
        # Cache the result
        _cache.set_outstanding_shares(ticker, date, float(shares))
            
        logger.debug(f"Final outstanding shares for {ticker} at {date}: {shares}", 
                    module="get_outstanding_shares", ticker=ticker)
        
        if verbose_data:
            logger.debug("=== OUTSTANDING SHARES DETAILS ===", module="get_outstanding_shares", ticker=ticker)
            logger.debug(f"Date requested: {date}", module="get_outstanding_shares", ticker=ticker)
            logger.debug(f"Quarter date used: {quarter_date.strftime('%Y-%m-%d')}", module="get_outstanding_shares", ticker=ticker)
            if relevant_splits:
                logger.debug(f"Applied {len(relevant_splits)} stock splits between {q_date_str} and {date}", 
                           module="get_outstanding_shares", ticker=ticker)

            
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
        if market_cap is None or end_date != datetime.datetime.now().strftime('%Y-%m-%d'):
            market_cap = get_historical_market_cap(ticker, end_date, verbose_data)
            logger.info(f"Market cap for{[ticker]} is either None or the end_date isn't the current date.",
                         module="get_market_cap")
        
        return market_cap
    except Exception as e:
        logger.error(f"Error fetching market cap for {ticker}: {e}", 
                    module="get_market_cap", ticker=ticker)
        return None


def get_historical_market_cap(ticker: str, date: str, verbose_data: bool = False) -> Optional[float]:
    """
    Get historical market cap for a ticker at a specific date by retrieving shares outstanding 
    from the most recent quarterly balance sheet and adjusting for any stock splits that 
    occurred between that quarter's end date and the requested date.
    """

    logger.debug(f"Running get_historical_market_cap() for {ticker} at {date}", 
                module="get_historical_market_cap", ticker=ticker)

    try:
        
        end_date = datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=1) # Since end_date in yfinance is exclusive
        end_date_str = end_date.strftime('%Y-%m-%d')

        ticker_obj = yf.Ticker(ticker)

        # Get outstanding shares
        shares = get_outstanding_shares(ticker, date, verbose_data)
        
        if not shares:
            logger.warning(f"Could not determine shares outstanding for {ticker} at {date}", 
                          module="get_historical_market_cap", ticker=ticker)
            return None
            

        # Get historical price
        hist = ticker_obj.history(start=date, end=end_date_str)
        logger.debug(f"Historical price for {[ticker]}: {hist} (1d {date} -> {end_date_str})", 
                     module="get_historical_market_cap", ticker=ticker)
        
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
    ticker: str, # Review: Since there is only a string here, does it mean it loads the news individually and the setting up a loop doesnt bring any advantage?
    end_date: str,
    start_date: str = None,
    limit: int = 1000,
    verbose_data: bool = False,
) -> List[InsiderTrade]:
    """
    Fetch insider trades for a single ticker.
    Maintains compatibility with the original function.
    """

    logger.debug(f"Running get_insider_trades() for {ticker} from {start_date or 'earliest'} to {end_date}", 
                module="get_insider_trades", ticker=ticker)

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(get_insider_trades_async([ticker], end_date, start_date, verbose_data))
        return results.get(ticker, [])[:limit]
    finally:
        loop.close()


async def get_insider_trades_async(tickers: List[str], end_date: str, start_date: str = None, verbose_data: bool = False) -> Dict[str, List[InsiderTrade]]:
    """
    Fetch insider trades for multiple tickers from SEC EDGAR.
    
    Args:
        tickers: List of stock ticker symbols
        end_date: End date string (YYYY-MM-DD)
        start_date: Optional start date (YYYY-MM-DD)
        
    Returns:
        Dictionary mapping tickers to lists of InsiderTrade objects
    """

    logger.debug(f"Running get_insider_trades_async() for {len(tickers)} tickers", 
                module="get_insider_trades_async")

    results = {}

    # Cache tracking
    cache_hits = 0
    total_tickers = len(tickers)
    
    # First try to get data from cache
    for ticker in tickers:
        cached_data = _cache.get_insider_trades(ticker)
        if cached_data:
            # Filter by date range
            filtered_data = [
                InsiderTrade(**trade) for trade in cached_data # Review: Why are there two** in InsiderTrade(**trade)?
                if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                and (trade.get("transaction_date") or trade["filing_date"]) <= end_date
            ]
            
            # If we have recent data, use it
            if filtered_data and (datetime.datetime.now() - datetime.datetime.strptime(cached_data[-1]["filing_date"], "%Y-%m-%d")).days < 7:
                results[ticker] = filtered_data
                cache_hits += 1
                continue
    
    # For any tickers not found in cache, fetch from SEC EDGAR
    tickers_to_fetch = [ticker for ticker in tickers if ticker not in results]
    
    # Debug: Log cache statistics
    if verbose_data:
        cache_percentage = (cache_hits / total_tickers * 100) if total_tickers > 0 else 0
        download_percentage = 100 - cache_percentage
        logger.debug(f"INSIDER TRADES CACHE STATS:", module="get_insider_trades_async")    
        logger.debug(f"  - From cache: {cache_hits}/{total_tickers} tickers ({cache_percentage:.1f}%)", module="get_insider_trades_async")    
        logger.debug(f"  - To download: {len(tickers_to_fetch)}/{total_tickers} tickers ({download_percentage:.1f}%)", module="get_insider_trades_async")

    if tickers_to_fetch:
        try:
            # Use the SEC EDGAR scraper to fetch insider trades
            edgar_results = await fetch_multiple_insider_trades(tickers_to_fetch, start_date, end_date)
            logger.debug(f"SEC EDGAR scraper results: {edgar_results}",
                         module="get_insider_trades_async")
            
            # Update results and cache
            for ticker, trades in edgar_results.items():
                # Cache the trades
                if trades:
                    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in trades])
                
                # Add to results
                results[ticker] = trades
                
        except Exception as e:
            logger.error(f"Error fetching insider trades from SEC EDGAR: {e}", 
                        module="get_insider_trades_async")
            # Fall back to returning empty lists for tickers not in cache
            for ticker in tickers_to_fetch:
                results[ticker] = []
    
    # Debug: Data preview before return
    if verbose_data:
        if cache_hits > 0:
            for ticker, trades in results.items():
                if len(trades) > 0 and ticker not in tickers_to_fetch:
                    logger.debug(f"Sample cached insider trades for {ticker}:" , module="get_insider_trades_async", ticker=ticker)
                    logger.debug(f"Total trades: {len(trades)}" , module="get_insider_trades_async", ticker=ticker)
                    logger.debug(f"Most recent trade: {trades[0].filing_date}", module="get_insider_trades_async", ticker=ticker)
                    logger.debug(f"Transaction shares: {trades[0].transaction_shares}, Value: {trades[0].transaction_value}", module="get_insider_trades_async", ticker=ticker)
                    
                    break
                else:
                    logger.debug("Preview of cached data not working", module="get_insider_trades_async", ticker=ticker)
        else:
            logger.debug("No cached data to preview", module="get_insider_trades_async")
        
        # Debug: Preview downloaded data
        if tickers_to_fetch and any(ticker in results for ticker in tickers_to_fetch):
            for ticker in tickers_to_fetch:
                if ticker in results and results[ticker]:
                    logger.debug(f"Sample downloaded insider trades for {ticker}:" , module="get_insider_trades_async", ticker=ticker)
                    logger.debug(f"Total trades: {len(results[ticker])}" , module="get_insider_trades_async", ticker=ticker)
                    logger.debug(f"Most recent trade: {results[ticker][0].filing_date if results[ticker] else 'N/A'}" , module="get_insider_trades_async", ticker=ticker)
                    logger.debug(f"Transaction shares: {results[ticker][0].transaction_shares if results[ticker] else 'N/A'}, " , module="get_insider_trades_async", ticker=ticker)
                    logger.debug(f"Value: {results[ticker][0].transaction_value if results[ticker] else 'N/A'}"  , module="get_insider_trades_async", ticker=ticker)

                    break
                else:
                    logger.debug("Preview of downloaded data not working", module="get_insider_trades_async", ticker=ticker)
        else:
            logger.debug("No downloaded data to preview", module="get_insider_trades_async")


    return results


# ===== COMPANY NEWS =====
def get_company_news(
    ticker: str, # Review: Since there is only a string here, does it mean it loads the news individually and the setting up a loop doesnt bring any advantage?
    end_date: str,
    start_date: str = None,
    limit: int = 1000,
    verbose_data: bool = False
) -> List[CompanyNews]:
    """
    Fetch company news for a single ticker.
    Maintains compatibility with the original function.
    """

    from utils.logger import logger

    logger.debug(f"Running get_company_news() for {[ticker]} from {start_date} to {end_date}", module="get_company_news", ticker=ticker)

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(get_company_news_async([ticker], end_date, start_date, verbose_data))
        return results.get(ticker, [])[:limit]
    finally:
        loop.close()


async def get_company_news_async(tickers: List[str], end_date: str, start_date: str = None, verbose_data: bool = False) -> Dict[str, List[CompanyNews]]:
    """
    Fetch company news from Yahoo Finance.
    This is a minimal implementation that could be expanded with sentiment analysis.
    
    Args:
        tickers: List of stock ticker symbols
        end_date: End date string (YYYY-MM-DD)
        start_date: Optional start date (YYYY-MM-DD)
        
    Returns:
        Dictionary mapping tickers to lists of CompanyNews objects
    """

    logger.debug(f"Running get_company_news_async() for {[tickers]} from {start_date} to {end_date}", module="get_company_news_async")

    results = {}

    # Cache tracking
    cache_hits = 0
    total_tickers = len(tickers)
    
    # First try to get data from cache
    for ticker in tickers:
        cached_data = _cache.get_company_news(ticker)
        if cached_data:
            # Filter by date range
            filtered_data = [
                CompanyNews(**news) for news in cached_data
                if (start_date is None or news["date"] >= start_date)
                and news["date"] <= end_date
            ]
            
            # If we have recent data, use it
            if filtered_data and (datetime.datetime.now() - datetime.datetime.strptime(cached_data[-1]["date"], "%Y-%m-%d")).days < 3:
                results[ticker] = filtered_data
                cache_hits += 1
                continue
    
    # For any tickers not found in cache or with outdated data, fetch from Yahoo Finance
    tickers_to_fetch = [ticker for ticker in tickers if ticker not in results]
    
    # Log cache statistics
    if verbose_data:
        cache_percentage = (cache_hits / total_tickers * 100) if total_tickers > 0 else 0
        download_percentage = 100 - cache_percentage
        logger.debug(f"COMPANY NEWS CACHE STATS:", module="get_company_news_async")
        logger.debug(f"  - From cache: {cache_hits}/{total_tickers} tickers ({cache_percentage:.1f}%)", module="get_company_news_async")   
        logger.debug(f"  - To download: {len(tickers_to_fetch)}/{total_tickers} tickers ({download_percentage:.1f}%)", module="get_company_news_async")
        
    
    if tickers_to_fetch:
        for ticker in tickers_to_fetch:
            try:
                # with LogCapture(debug_mode=True, module="yfinance") as capture:
                ticker_obj = yf.Ticker(ticker)
                news_data = ticker_obj.news
                
                news_list = []
                for news_item in news_data:
                    # Convert unix timestamp to date
                    timestamp = news_item.get('providerPublishTime', 0)
                    date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                    
                    # Filter by date range
                    if (start_date is None or date >= start_date) and date <= end_date:
                        news = CompanyNews(
                            ticker=ticker,
                            title=news_item.get('title', ''),
                            author=news_item.get('publisher', ''),
                            source=news_item.get('publisher', ''),
                            date=date,
                            url=news_item.get('link', ''),
                            sentiment=None  # Would need sentiment analysis
                        )
                        news_list.append(news)
                
                # Cache the news
                if news_list:
                    _cache.set_company_news(ticker, [news.model_dump() for news in news_list])
                
                # Add to results
                results[ticker] = news_list
                
            except Exception as e:
                print(f"Error fetching news for {ticker}: {e}")
                results[ticker] = []
    
    # Debug: Data preview before return
    if verbose_data: 
    # Preview cached data
        if cache_hits > 0:
            for ticker, news_items in results.items():
                if len(news_items) > 0 and ticker not in tickers_to_fetch:
                    logger.debug(f"Sample cached company news for {ticker}:", module="get_company_news_async", ticker=ticker)
                    logger.debug(f"Total news items: {len(news_items)}", module="get_company_news_async", ticker=ticker)
                    logger.debug(f"Most recent news ({news_items[0].date}): {news_items[0].title}", module="get_company_news_async", ticker=ticker)
                    
                    break
        
        # Preview downloaded data
        if tickers_to_fetch and any(ticker in results for ticker in tickers_to_fetch):
            for ticker in tickers_to_fetch:
                if ticker in results and results[ticker]:
                    logger.debug(f"Sample downloaded company news for {ticker}:", module="get_company_news_async", ticker=ticker)
                    logger.debug(f"Total news items: {len(results[ticker])}", module="get_company_news_async", ticker=ticker)
                    logger.debug(f"Most recent news ({results[ticker][0].date if results[ticker] else 'N/A'}): " , module="get_company_news_async", ticker=ticker)
                        
                    break
    
    return results # Review: Do I need to use return results.get(ticker, [])
    

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
    Fetch specific line items from financial statements.
    Reimplemented to use yfinance.
    
    Args:
        ticker: Stock ticker symbol
        line_items: List of line items to retrieve
        end_date: End date string (YYYY-MM-DD)
        period: Period type ("ttm", "annual", etc.)
        limit: Maximum number of periods to return
        
    Returns:
        List of LineItem objects
    """

    logger.debug(f"Running search_line_items() for {ticker}: {line_items}", 
                module="search_line_items", ticker=ticker)

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
        
        # Process each date column in the statements
        for i, col in enumerate(income_stmt.columns):
            if i >= limit:
                break
                
            report_date = col.strftime('%Y-%m-%d')
            if report_date > end_date:
                continue
                
            # Create a line item for this date
            line_item = LineItem(
                ticker=ticker,
                report_period=report_date,
                period=period,
                currency="USD"
            )

            if verbose_data:
                logger.debug(f"report date: {report_date}, end_date: {end_date}", module="search_line_items", ticker=ticker)
            
            # Add requested line items
            for item in line_items:
                # Map line_items to yfinance fields
                item_mapping = {
                    "revenue": ("Total Revenue", income_stmt),
                    "net_income": ("Net Income", income_stmt),
                    "capital_expenditure": ("Capital Expenditure", cash_flow),
                    "depreciation_and_amortization": ("Depreciation And Amortization", cash_flow),
                    "outstanding_shares": None,  # Special handling
                    "total_assets": ("Total Assets", balance_sheet),
                    "total_liabilities": ("Total Liabilities Net Minority Interest", balance_sheet),
                    "working_capital": None,  # Special calculation 
                    "debt_to_equity": None,  # Special calculation
                    "free_cash_flow": None,  # Special calculation
                    "operating_margin": None,  # Special calculation
                    "dividends_and_other_cash_distributions": ("Dividends Paid", cash_flow),
                }
                
                if item == "outstanding_shares":
                    # Use the centralized get_outstanding_shares function
                    shares = get_outstanding_shares(ticker, end_date, verbose_data=verbose_data) 
                    if shares:
                        setattr(line_item, item, float(shares))
                        logger.debug(f"Successfully set outstanding_shares={shares} for {ticker}", 
                                    module="search_line_items", ticker=ticker)
                    else:
                        logger.warning(f"Cannot set outstanding_shares: no data available for {ticker} on {report_date}", 
                                    module="search_line_items", ticker=ticker)
                
                elif item == "working_capital":
                    # Working capital = Current Assets - Current Liabilities
                    current_assets = balance_sheet.loc["Total Current Assets", col] if "Total Current Assets" in balance_sheet.index else None
                    current_liabilities = balance_sheet.loc["Total Current Liabilities", col] if "Total Current Liabilities" in balance_sheet.index else None
                    if current_assets is not None and current_liabilities is not None:
                        setattr(line_item, item, float(current_assets - current_liabilities))
                    else:
                        logger.warning(f"Missing data for working_capital calculation: current_assets={current_assets}, current_liabilities={current_liabilities}", 
                        module="search_line_items", ticker=ticker)
                
                elif item == "debt_to_equity":
                    # Debt to Equity
                    total_liabilities = balance_sheet.loc["Total Liabilities Net Minority Interest", col] if "Total Liabilities Net Minority Interest" in balance_sheet.index else None
                    stockholders_equity = balance_sheet.loc["Stockholders Equity", col] if "Stockholders Equity" in balance_sheet.index else None
                    if total_liabilities is not None and stockholders_equity is not None and float(stockholders_equity) > 0:
                        setattr(line_item, item, float(total_liabilities) / float(stockholders_equity))
                    else:
                        logger.warning(f"Missing data for debt_to_equity calculation: total_liabilities={total_liabilities}, stockholders_equity={stockholders_equity}", 
                        module="search_line_items", ticker=ticker)
                
                elif item == "free_cash_flow":
                    # Free Cash Flow = Operating Cash Flow + Capital Expenditure
                    operating_cash_flow = cash_flow.loc["Operating Cash Flow", col] if "Operating Cash Flow" in cash_flow.index else None
                    capital_expenditure = cash_flow.loc["Capital Expenditure", col] if "Capital Expenditure" in cash_flow.index else None
                    if operating_cash_flow is not None and capital_expenditure is not None:
                        setattr(line_item, item, float(operating_cash_flow + capital_expenditure))
                    else:
                        logger.warning(f"Missing data for free_cash_flow calculation: operating_cash_flow={operating_cash_flow}, capital_expenditure={capital_expenditure}", 
                        module="search_line_items", ticker=ticker)
                
                elif item == "operating_margin":
                    # Operating Margin = Operating Income / Revenue
                    operating_income = income_stmt.loc["Operating Income", col] if "Operating Income" in income_stmt.index else None
                    total_revenue = income_stmt.loc["Total Revenue", col] if "Total Revenue" in income_stmt.index else None
                    if operating_income is not None and total_revenue is not None and float(total_revenue) > 0:
                        setattr(line_item, item, float(operating_income) / float(total_revenue))
                    else:
                        logger.warning(f"Missing data for operating_margin calculation: operating_income={operating_income}, total_revenue={total_revenue}", 
                        module="search_line_items", ticker=ticker)
                
                elif item_mapping.get(item):
                    field, df = item_mapping[item]
                    if field in df.index:
                        value = df.loc[field, col]
                        setattr(line_item, item, float(value))

            logger.debug(f"Retrieved {len(results)} line items for {ticker}", 
                    module="search_line_items", ticker=ticker)
            
            results.append(line_item)
        
        return results
    
    except Exception as e:
        logger.error(f"Error searching line items for {ticker}: {e}", 
                module="search_line_items", ticker=ticker)
        return []


# ===== LIKELY A USELESS FUNCTION =====
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
        
    Returns:
        Dictionary of ticker to data mapping
    """
    
    logger.debug(f"Running get_data_for_tickers() for {len(tickers)} tickers from {start_date} to {end_date}", 
                module="get_data_for_tickers")

    # Split tickers into batches of batch_size
    ticker_batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    
    results = {}
    loop = asyncio.new_event_loop()
    
    try:
        asyncio.set_event_loop(loop)
        
        # Process each batch of tickers
        for batch in ticker_batches:
            # Get price data, financial metrics, insider trades, and news in a batch
            price_task = fetch_prices_batch(batch, start_date, end_date, verbose_data)
            metrics_task = fetch_financial_metrics_async(batch, end_date, verbose_data)
            insider_task = get_insider_trades_async(batch, end_date, start_date, verbose_data)
            news_task = get_company_news_async(batch, end_date, start_date, verbose_data)
            
            # Run all tasks concurrently
            batch_results = loop.run_until_complete(
                asyncio.gather(price_task, metrics_task, insider_task, news_task)
            )
            
            # Process results for this batch
            price_data, metrics_data, insider_data, news_data = batch_results
            
            # Merge into results dictionary
            for ticker in batch:
                results[ticker] = {
                    "prices": price_data.get(ticker, pd.DataFrame()),
                    "metrics": metrics_data.get(ticker, []),
                    "insider_trades": insider_data.get(ticker, []),
                    "news": news_data.get(ticker, [])
                }
    except Exception as e:
        logger.error(f"Error fetching data for tickers: {e}", module="get_data_for_tickers")
    finally:
        loop.close()

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

