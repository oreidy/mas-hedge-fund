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

# Import the market calendar and SEC EDGAR scraper
from market_calendar import is_trading_day, adjust_date_range
from sec_edgar_scraper import fetch_insider_trades_for_ticker, fetch_multiple_insider_trades

# ===== CACHE INITIALIZATION =====
# Create a global cache instance with disk persistence
_cache = DiskCache(cache_dir=Path("./cache"))


# ===== PRICE FUNCTIONS =====
def get_prices(ticker: str, start_date: str, end_date: str) -> List[Price]:
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

    # Ensure dates are strings
    start_date = str(start_date) if not isinstance(start_date, str) else start_date
    end_date = str(end_date) if not isinstance(end_date, str) else end_date

    # Check if dates are trading days and adjust if needed
    try:
        if not is_trading_day(start_date) or not is_trading_day(end_date):
            adjusted_start, adjusted_end = adjust_date_range(start_date, end_date)
            print(f"Adjusting date range for {ticker}: {start_date}-{end_date} -> {adjusted_start}-{adjusted_end}")
            start_date, end_date = adjusted_start, adjusted_end
    except ValueError as e:
        print(f"Warning: {e} - continuing with original dates")

    # Run the async function in a new event loop
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(get_prices_async([ticker], start_date, end_date))
        return results.get(ticker, [])
    finally:
        loop.close()


async def get_prices_async(tickers: List[str], start_date: str, end_date: str) -> Dict[str, List[Price]]:
    """Fetch prices for multiple tickers asynchronously"""

    # Ensure dates are strings
    start_date = str(start_date) if not isinstance(start_date, str) else start_date
    end_date = str(end_date) if not isinstance(end_date, str) else end_date

    # Check if dates need adjustment for trading days
    if not is_trading_day(start_date) or not is_trading_day(end_date):
        try:
            start_date, end_date = adjust_date_range(start_date, end_date)
        except ValueError as e:
            print(f"Warning: {e} - continuing with original dates")

    df_results = await fetch_prices_batch(tickers, start_date, end_date)
    return {ticker: df_to_price_objects(df) for ticker, df in df_results.items()}


async def fetch_prices_batch(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical prices for multiple tickers in a single batch request.
    Uses yfinance's batch download capability for efficiency.
    Adjusts dates to trading days to handle market holidays.
    """
    # First check cache for all tickers
    cached_results = {}
    tickers_to_fetch = []
    
    for ticker in tickers:
        cached_data = _cache.get_prices(ticker, start_date, end_date)
        if cached_data:
            # Convert cached data to DataFrame
            df = pd.DataFrame(cached_data)
            df["Date"] = pd.to_datetime(df["time"])
            df.set_index("Date", inplace=True)
            cached_results[ticker] = df
        else:
            tickers_to_fetch.append(ticker)
    
    if not tickers_to_fetch:
        return cached_results
    
    # Fetch data for tickers not in cache
    try:
        # yfinance batch download
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
        
        # Handle single vs multiple ticker results differently
        if len(tickers_to_fetch) == 1:
            ticker = tickers_to_fetch[0]
            # For single ticker, data is a simple DataFrame
            df = data.copy()
            df.columns = [col.lower() for col in df.columns]
            df.rename(columns={'adj close': 'close'}, inplace=True)
            
            # Cache the processed data
            _cache.set_prices(ticker, df_to_cache_format(df))
            results[ticker] = df
        else:
            # For multiple tickers, data is a MultiIndex DataFrame
            for ticker in tickers_to_fetch:
                if ticker in data.columns.levels[0]:
                    ticker_data = data[ticker].copy()
                    ticker_data.columns = [col.lower() for col in ticker_data.columns]
                    if 'adj close' in ticker_data.columns:
                        ticker_data.rename(columns={'adj close': 'close'}, inplace=True)
                    
                    # Cache the processed data
                    _cache.set_prices(ticker, df_to_cache_format(ticker_data))
                    results[ticker] = ticker_data
        
        return results
        
    except Exception as e:
        print(f"Error fetching batch price data: {e}")
        return cached_results


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get price data as DataFrame. Same as original function."""
    prices = get_prices(ticker, start_date, end_date)
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
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(fetch_financial_metrics_async([ticker], end_date, period, limit))
        return results.get(ticker, [])
    except Exception as e:
        print(f"Error fetching prices for {ticker} between {start_date} and {end_date}: {e}")
    finally:
        loop.close()


async def fetch_financial_metrics_async(tickers: List[str], end_date: str, period: str = "ttm", limit: int = 10):
    """Fetch financial metrics for multiple tickers in parallel"""
    async def fetch_ticker_metrics(ticker):
        cached_data = _cache.get_financial_metrics(ticker)
        if cached_data:
            filtered_data = [metric for metric in cached_data if metric["report_period"] <= end_date]
            filtered_data.sort(key=lambda x: x["report_period"], reverse=True)
            if filtered_data:
                return ticker, [FinancialMetrics(**metric) for metric in filtered_data[:limit]]
        
        # Not in cache, fetch from yfinance
        try:
            financial_metrics = []
            ticker_obj = yf.Ticker(ticker)
            
            # Get financials (income statement, balance sheet and cash flow statement)
            income_stmt = ticker_obj.income_stmt
            balance_sheet = ticker_obj.balance_sheet
            cash_flow = ticker_obj.cashflow
            
            if income_stmt.empty or balance_sheet.empty or cash_flow.empty:
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
                    market_cap = get_historical_market_cap(ticker, report_date)
                    
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
                        enterprise_value_to_ebitda_ratio=None,
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
                    print(f"Error processing financial data for {ticker} at {report_date}: {e}")
                    continue
            
            # Cache the results
            if financial_metrics:
                _cache.set_financial_metrics(ticker, [metric.model_dump() for metric in financial_metrics])
            
            return ticker, financial_metrics
        except Exception as e:
            print(f"Error fetching financial metrics for {ticker}: {e}")
            return ticker, []
    
    # Create tasks for all tickers
    tasks = [fetch_ticker_metrics(ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    
    # Convert results to dictionary
    return {ticker: metrics for ticker, metrics in results}


# ===== MARKET CAP FUNTIONS =====
def get_market_cap(
    ticker: str,
    end_date: str,
) -> float:
    """Fetch market cap for a ticker at a specific date"""
    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Try using info first (current market cap, not historical)
        market_cap = ticker_obj.info.get('marketCap')
        
        # If we need a historical market cap, calculate it
        if market_cap is None or end_date != datetime.datetime.now().strftime('%Y-%m-%d'):
            market_cap = get_historical_market_cap(ticker, end_date)
        
        return market_cap
    except Exception as e:
        print(f"Error fetching market cap for {ticker}: {e}")
        return None


def get_historical_market_cap(ticker: str, date: str) -> Optional[float]:
    """Get historical market cap for a ticker at a specific date"""
    try:
        # Get historical price data
        end_date = datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=1)
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        ticker_obj = yf.Ticker(ticker)
        shares_outstanding = ticker_obj.info.get('sharesOutstanding')
        
        if not shares_outstanding:
            return None
        
        # Get historical price
        hist = ticker_obj.history(start=date, end=end_date_str)
        if hist.empty:
            return None
        
        close_price = hist['Close'].iloc[0]
        return close_price * shares_outstanding
    except Exception as e:
        print(f"Error calculating historical market cap for {ticker} at {date}: {e}")
        return None


# ===== INSIDER TRADES FROM SEC EDGAR =====
def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str = None,
    limit: int = 1000,
) -> List[InsiderTrade]:
    """
    Fetch insider trades for a single ticker.
    Maintains compatibility with the original function.
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(get_insider_trades_async([ticker], end_date, start_date))
        return results.get(ticker, [])[:limit]
    finally:
        loop.close()


async def get_insider_trades_async(tickers: List[str], end_date: str, start_date: str = None) -> Dict[str, List[InsiderTrade]]:
    """
    Fetch insider trades for multiple tickers from SEC EDGAR.
    
    Args:
        tickers: List of stock ticker symbols
        end_date: End date string (YYYY-MM-DD)
        start_date: Optional start date (YYYY-MM-DD)
        
    Returns:
        Dictionary mapping tickers to lists of InsiderTrade objects
    """
    results = {}
    
    # First try to get data from cache
    for ticker in tickers:
        cached_data = _cache.get_insider_trades(ticker)
        if cached_data:
            # Filter by date range
            filtered_data = [
                InsiderTrade(**trade) for trade in cached_data
                if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                and (trade.get("transaction_date") or trade["filing_date"]) <= end_date
            ]
            
            # If we have recent data, use it
            if filtered_data and (datetime.datetime.now() - datetime.datetime.strptime(cached_data[-1]["filing_date"], "%Y-%m-%d")).days < 7:
                results[ticker] = filtered_data
                continue
    
    # For any tickers not found in cache, fetch from SEC EDGAR
    tickers_to_fetch = [ticker for ticker in tickers if ticker not in results]
    
    if tickers_to_fetch:
        try:
            # Use our SEC EDGAR scraper to fetch insider trades
            edgar_results = await fetch_multiple_insider_trades(tickers_to_fetch, start_date, end_date)
            
            # Update results and cache
            for ticker, trades in edgar_results.items():
                # Cache the trades
                if trades:
                    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in trades])
                
                # Add to results
                results[ticker] = trades
                
        except Exception as e:
            print(f"Error fetching insider trades from SEC EDGAR: {e}")
            # Fall back to returning empty lists for tickers not in cache
            for ticker in tickers_to_fetch:
                results[ticker] = []
    
    return results


# ===== COMPANY NEWS =====
def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str = None,
    limit: int = 1000,
) -> List[CompanyNews]:
    """
    Fetch company news for a single ticker.
    Maintains compatibility with the original function.
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(get_company_news_async([ticker], end_date, start_date))
        return results.get(ticker, [])[:limit]
    finally:
        loop.close()


async def get_company_news_async(tickers: List[str], end_date: str, start_date: str = None) -> Dict[str, List[CompanyNews]]:
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
    results = {}
    
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
                continue
    
    # For any tickers not found in cache or with outdated data, fetch from Yahoo Finance
    tickers_to_fetch = [ticker for ticker in tickers if ticker not in results]
    
    if tickers_to_fetch:
        for ticker in tickers_to_fetch:
            try:
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
    
    return results


# ===== USELESS FUNCTIONS =====
def search_line_items(
    ticker: str,
    line_items: list[str],
    start_date: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
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
                
                # Handle special calculations
                if item == "outstanding_shares":
                    # Approximate shares outstanding from market cap / price
                    market_cap = get_historical_market_cap(ticker, report_date)
                    if market_cap:
                        hist = ticker_obj.history(start=report_date, end=(datetime.datetime.strptime(report_date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
                        if not hist.empty:
                            price = hist['Close'].iloc[0]
                            if price > 0:
                                setattr(line_item, item, market_cap / price)
                
                elif item == "working_capital":
                    # Working capital = Current Assets - Current Liabilities
                    current_assets = balance_sheet.loc.get("Total Current Assets", col)
                    current_liabilities = balance_sheet.loc.get("Total Current Liabilities", col)
                    if current_assets is not None and current_liabilities is not None:
                        setattr(line_item, item, float(current_assets - current_liabilities))
                
                elif item == "debt_to_equity":
                    # Debt to Equity
                    total_liabilities = balance_sheet.loc.get("Total Liabilities Net Minority Interest", col)
                    stockholders_equity = balance_sheet.loc.get("Stockholders Equity", col)
                    if total_liabilities is not None and stockholders_equity is not None and float(stockholders_equity) > 0:
                        setattr(line_item, item, float(total_liabilities) / float(stockholders_equity))
                
                elif item == "free_cash_flow":
                    # Free Cash Flow = Operating Cash Flow + Capital Expenditure
                    operating_cash_flow = cash_flow.loc.get("Operating Cash Flow", col)
                    capital_expenditure = cash_flow.loc.get("Capital Expenditure", col)
                    if operating_cash_flow is not None and capital_expenditure is not None:
                        setattr(line_item, item, float(operating_cash_flow + capital_expenditure))
                
                elif item == "operating_margin":
                    # Operating Margin = Operating Income / Revenue
                    operating_income = income_stmt.loc.get("Operating Income", col)
                    total_revenue = income_stmt.loc.get("Total Revenue", col)
                    if operating_income is not None and total_revenue is not None and float(total_revenue) > 0:
                        setattr(line_item, item, float(operating_income) / float(total_revenue))
                
                elif item_mapping.get(item):
                    field, df = item_mapping[item]
                    if field in df.index:
                        value = df.loc[field, col]
                        setattr(line_item, item, float(value))
            
            results.append(line_item)
        
        return results
    
    except Exception as e:
        print(f"Error searching line items for {ticker}: {e}")
        return []


async def get_price_data_batch(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Get price data as DataFrames for multiple tickers in one batch request"""
    return await fetch_prices_batch(tickers, start_date, end_date)


def get_data_for_tickers(tickers: List[str], start_date: str, end_date: str, batch_size: int = 20):
    """
    Process multiple tickers efficiently with batching and parallel processing.
    Returns a dictionary with all data for each ticker.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        batch_size: Number of tickers to process in each batch
        
    Returns:
        Dictionary of ticker to data mapping
    """

    # Check for market holidays
    try:
        if not is_trading_day(start_date) or not is_trading_day(end_date):
            adjusted_start, adjusted_end = adjust_date_range(start_date, end_date)
            print(f"Adjusting date range for {ticker}: {start_date}-{end_date} -> {adjusted_start}-{adjusted_end}")
            start_date, end_date = adjusted_start, adjusted_end
    except ValueError as e:
        print(f"Warning: {e} - continuing with original dates")

    # Split tickers into batches of batch_size
    ticker_batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    
    results = {}
    loop = asyncio.new_event_loop()
    
    try:
        asyncio.set_event_loop(loop)
        
        # Process each batch of tickers
        for batch in ticker_batches:
            # Get price data in a batch
            price_task = fetch_prices_batch(batch, start_date, end_date)
            
            # Get financial metrics and news in parallel
            metrics_task = fetch_financial_metrics_async(batch, end_date)
            insider_task = get_insider_trades_async(batch, end_date, start_date)
            news_task = get_company_news_async(batch, end_date, start_date)
            
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
    
    finally:
        loop.close()
    
    return results

