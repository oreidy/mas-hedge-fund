import pandas as pd
import requests
from io import StringIO
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import wrds
except ImportError:
    wrds = None

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import logger
from data.disk_cache import DiskCache, CACHE_DIR

def get_nyse_tickers(include_etfs=False):
    """
    Downloads NYSE-listed tickers from Nasdaq's FTP and returns them as a list.
    
    Args:
        include_etfs (bool): If False, excludes ETFs from the list (default: False).
    
    Returns:
        list: NYSE-listed tickers (e.g., ['AAPL', 'MSFT', ...]).
    """
    # URL for the NYSE-listed securities CSV (hosted on datahub.io)
    url = "https://datahub.io/core/nyse-other-listings/r/nyse-listed.csv"
    
    try:
        # Download and read the CSV
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        data = StringIO(response.text)
        df = pd.read_csv(data)
        
        # Filter out ETFs if needed (column 'ETF' is 'Y' for ETFs)
        if not include_etfs and 'ETF' in df.columns:
            df = df[df['ETF'] != 'Y']
        
        # Extract tickers from the 'ACT Symbol' column
        tickers = df['ACT Symbol'].tolist()
        logger.info(f"Found {len(tickers)} NYSE tickers")
        return tickers
    
    except Exception as e:
        logger.warning(f"Error fetching NYSE tickers: {e}")
        return []


def get_sp500_constituents_wrds(years_back: int = 4) -> List[str]:
    """
    Gets all unique S&P 500 tickers from WRDS for the last specified years.
    Returns a list of all tickers that were in the S&P 500 for at least one day during the period.
    Does NOT cache results - always fetches fresh data from WRDS.
    
    Args:
        years_back (int): Number of years to look back (default: 4).
    
    Returns:
        List[str]: All unique tickers that were in S&P 500 during the period.
    """
    if wrds is None:
        logger.warning("WRDS not available. Install with: pip install wrds")
        return []
    
    try:
        # Connect to WRDS using credentials from environment variables
        username = os.getenv('WRDS_USERNAME')
        password = os.getenv('WRDS_PASSWORD')
        
        if not username or not password:
            logger.warning("WRDS credentials not found in environment variables")
            return []
            
        conn = wrds.Connection(wrds_username=username)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        # Debug: log the date range
        logger.info(f"Fetching S&P 500 tickers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Improved SQL query to get all unique historical tickers that had ANY overlap with our period
        historical_tickers_query = f"""
            SELECT DISTINCT d.ticker, s.start as sp500_start, s.ending as sp500_end
            FROM crsp.dsp500list s
            JOIN crsp.dsenames d ON s.permno = d.permno
            WHERE (
                -- S&P 500 membership overlaps with our date range
                s.start <= '{end_date.strftime("%Y-%m-%d")}'
                AND (s.ending IS NULL OR s.ending >= '{start_date.strftime("%Y-%m-%d")}')
            )
            AND (
                -- Ticker name was valid during S&P membership
                d.namedt <= COALESCE(s.ending, '{end_date.strftime("%Y-%m-%d")}')
                AND d.nameendt >= s.start
            )
            AND d.ticker IS NOT NULL
            AND d.ticker != ''
            ORDER BY d.ticker
        """
        
        result = conn.raw_sql(historical_tickers_query)
        
        # Debug: show some sample results
        logger.info(f"Raw query returned {len(result)} records")
        if len(result) > 0:
            logger.info(f"Sample records:")
            for i in range(min(5, len(result))):
                row = result.iloc[i]
                logger.info(f"  {row['ticker']}: {row['sp500_start']} to {row['sp500_end']}")
        
        tickers = result['ticker'].tolist()
        
        # Normalize tickers for yfinance compatibility
        tickers = [normalize_ticker(ticker) for ticker in tickers]
        
        # Remove duplicates and sort
        tickers = sorted(list(set(tickers)))
        
        logger.info(f"Retrieved {len(tickers)} unique S&P 500 historical tickers from WRDS")
        return tickers
        
    except Exception as e:
        logger.warning(f"Error fetching S&P 500 constituents from WRDS: {e}")
        return []


def get_sp500_tickers_with_dates(years_back: int = 4) -> Dict[str, Dict]:
    """
    Gets S&P 500 tickers with their listing date information from cache or WRDS.
    Returns a dictionary mapping ticker to its S&P 500 membership dates.
    
    Args:
        years_back (int): Number of years to look back (default: 4).
    
    Returns:
        Dict[str, Dict]: {ticker: {'sp500_start': datetime, 'sp500_end': datetime|None}}
    """
    # Try to get from local cache first
    cache = DiskCache()
    cached_data = cache.get_sp500_membership_data(years_back)
    if cached_data:
        logger.debug(f"Retrieved S&P 500 membership data from cache for {len(cached_data)} tickers", 
                    module="get_sp500_tickers_with_dates")
        return cached_data
    
    # Cache miss - fetch from WRDS
    logger.info("S&P 500 membership data not in cache, fetching from WRDS...", 
               module="get_sp500_tickers_with_dates")
    
    if wrds is None:
        logger.warning("WRDS not available. Install with: pip install wrds")
        return {}
    
    try:
        # Connect to WRDS using credentials from environment variables
        username = os.getenv('WRDS_USERNAME')
        password = os.getenv('WRDS_PASSWORD')
        
        if not username or not password:
            logger.warning("WRDS credentials not found in environment variables")
            return {}
            
        conn = wrds.Connection(wrds_username=username)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        # Query to get tickers with their S&P 500 membership dates
        query = f"""
            SELECT DISTINCT d.ticker, s.start as sp500_start, s.ending as sp500_end
            FROM crsp.dsp500list s
            JOIN crsp.dsenames d ON s.permno = d.permno
            WHERE (
                -- S&P 500 membership overlaps with our date range
                s.start <= '{end_date.strftime("%Y-%m-%d")}'
                AND (s.ending IS NULL OR s.ending >= '{start_date.strftime("%Y-%m-%d")}')
            )
            AND (
                -- Ticker name was valid during S&P membership
                d.namedt <= COALESCE(s.ending, '{end_date.strftime("%Y-%m-%d")}')
                AND d.nameendt >= s.start
            )
            AND d.ticker IS NOT NULL
            AND d.ticker != ''
            ORDER BY d.ticker, s.start
        """
        
        result = conn.raw_sql(query)
        
        # Process results into dictionary format
        ticker_dates = {}
        for _, row in result.iterrows():
            ticker = normalize_ticker(row['ticker'])
            sp500_start = pd.to_datetime(row['sp500_start'])
            sp500_end = pd.to_datetime(row['sp500_end']) if row['sp500_end'] is not None else None
            
            # If ticker already exists, keep the earliest start date and latest end date
            if ticker in ticker_dates:
                existing = ticker_dates[ticker]
                sp500_start = min(sp500_start, existing['sp500_start'])
                if sp500_end is None or existing['sp500_end'] is None:
                    sp500_end = None  # Still active
                else:
                    sp500_end = max(sp500_end, existing['sp500_end'])
            
            ticker_dates[ticker] = {
                'sp500_start': sp500_start,
                'sp500_end': sp500_end
            }
        
        logger.debug(f"Retrieved S&P 500 membership data for {len(ticker_dates)} tickers from WRDS")
        
        # Apply manual fixes for 2025 S&P 500 changes
        ticker_dates = apply_manual_sp500_fixes(ticker_dates)
        
        # Cache the data for future use
        cache.set_sp500_membership_data(ticker_dates, years_back)
        logger.debug(f"Cached S&P 500 membership data for {len(ticker_dates)} tickers", 
                    module="get_sp500_tickers_with_dates")
        
        return ticker_dates
        
    except Exception as e:
        logger.error(f"Error fetching S&P 500 ticker dates from WRDS: {e}")
        return {}



def apply_manual_sp500_fixes(ticker_dates: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Apply manual fixes for 2025 S&P 500 membership changes.
    
    WRDS data has all tickers ending at 2024-12-31, but many are still in the index.
    This function sets sp500_end=None for currently active tickers and sets specific
    end dates for tickers that were actually removed in 2025.
    
    2025 S&P 500 Changes:
    - July 9, 2025: DDOG (Datadog) replaced JNPR (Juniper Networks) - HPE acquired Juniper
    - May 19, 2025: COIN (Coinbase) replaced DFS (Discover Financial) - Capital One acquired Discover
    - March 24, 2025: DASH (DoorDash) replaced BWA (BorgWarner) - Market cap change
    - March 24, 2025: TKO (TKO Group Holdings) replaced TFX (Teleflex) - Market cap change  
    - March 24, 2025: WSM (Williams-Sonoma) replaced CE (Celanese) - Market cap change
    - March 24, 2025: EXE (Expand Energy) replaced FMC (FMC Corporation) - Market cap change
    
    Args:
        ticker_dates: Dictionary of ticker membership data from WRDS
        
    Returns:
        Updated ticker_dates with manual fixes applied
    """
    # Tickers removed from S&P 500 in 2025 with their actual removal dates
    removed_tickers_2025 = {
        'JNPR': datetime(2025, 7, 9),   # Juniper Networks - acquired by HPE
        'DFS': datetime(2025, 5, 19),   # Discover Financial - acquired by Capital One
        'BWA': datetime(2025, 3, 24),   # BorgWarner - market cap change
        'TFX': datetime(2025, 3, 24),   # Teleflex - market cap change
        'CE': datetime(2025, 3, 24),    # Celanese - market cap change
        'FMC': datetime(2025, 3, 24),   # FMC Corporation - market cap change
    }
    
    # Count fixes applied
    fixed_active = 0
    fixed_removed = 0
    
    for ticker, dates in ticker_dates.items():
        # If ticker has sp500_end = 2024-12-31
        if (dates['sp500_end'] is not None and 
            dates['sp500_end'].strftime('%Y-%m-%d') == '2024-12-31'):
            
            if ticker in removed_tickers_2025:
                # Set actual removal date for tickers removed in 2025
                dates['sp500_end'] = removed_tickers_2025[ticker]
                fixed_removed += 1
                logger.debug(f"Fixed {ticker}: set sp500_end={removed_tickers_2025[ticker].strftime('%Y-%m-%d')} (actually removed)")
            else:
                # Set sp500_end to None for tickers still active
                dates['sp500_end'] = None
                fixed_active += 1
                logger.debug(f"Fixed {ticker}: set sp500_end=None (still active)")
    
    logger.debug(f"Applied manual S&P 500 fixes: {fixed_active} tickers set to active, "
               f"{fixed_removed} tickers updated with actual removal dates")
    
    return ticker_dates


def get_exchange_listing_date(ticker: str) -> Optional[datetime]:
    """
    Gets the stock exchange listing date (IPO date) for a ticker using yfinance.
    
    Args:
        ticker (str): Ticker symbol
        
    Returns:
        Optional[datetime]: IPO date if available, None if not found
    """
    import yfinance as yf
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try to get first trade date from info
        if 'firstTradeDateEpochUtc' in info and info['firstTradeDateEpochUtc']:
            import datetime as dt
            ipo_date = dt.datetime.fromtimestamp(info['firstTradeDateEpochUtc'])
            # Convert to naive datetime to avoid timezone issues
            if ipo_date.tzinfo is not None:
                ipo_date = ipo_date.replace(tzinfo=None)
            logger.debug(f"Found IPO date for {ticker}: {ipo_date.strftime('%Y-%m-%d')}")
            return ipo_date
        
        # Fallback: try to get from history (earliest available date)
        try:
            hist = stock.history(period="max")
            if not hist.empty:
                earliest_date = hist.index[0].to_pydatetime()
                # Convert to naive datetime to avoid timezone issues
                if earliest_date.tzinfo is not None:
                    earliest_date = earliest_date.replace(tzinfo=None)
                logger.debug(f"Using earliest price date for {ticker}: {earliest_date.strftime('%Y-%m-%d')}")
                return earliest_date
        except Exception:
            pass
            
        logger.debug(f"No IPO date found for {ticker}")
        return None
        
    except Exception as e:
        logger.debug(f"Error fetching IPO date for {ticker}: {e}")
        return None


def get_exchange_listing_dates(tickers: List[str]) -> Dict[str, datetime]:
    """
    Gets stock exchange listing dates (IPO dates) for multiple tickers with caching.
    
    Args:
        tickers (List[str]): List of ticker symbols
        
    Returns:
        Dict[str, datetime]: Mapping of ticker to IPO date (only includes tickers with valid dates)
    """
    # Initialize cache for IPO dates
    cache = DiskCache()
    
    # Add IPO dates cache type with 30 day refresh (IPO dates don't change)
    if "ipo_dates" not in cache.ttls:
        from diskcache import Cache
        cache.ttls["ipo_dates"] = timedelta(days=30)
        cache.caches["ipo_dates"] = Cache(CACHE_DIR / "ipo_dates")
    
    ipo_dates = {}
    tickers_to_fetch = []
    
    # Check cache first
    for ticker in tickers:
        cached_date = cache.caches["ipo_dates"].get(f"ipo:{ticker}")
        if cached_date:
            ipo_dates[ticker] = cached_date
        else:
            tickers_to_fetch.append(ticker)
    
    if tickers_to_fetch:
        logger.info(f"Fetching IPO dates for {len(tickers_to_fetch)} tickers (cache miss)")
        
        # Fetch missing IPO dates
        for ticker in tickers_to_fetch:
            ipo_date = get_exchange_listing_date(ticker)
            if ipo_date:
                ipo_dates[ticker] = ipo_date
                # Cache the result
                cache.caches["ipo_dates"].set(
                    f"ipo:{ticker}", 
                    ipo_date, 
                    expire=cache._get_expiration("ipo_dates")
                )
    
    return ipo_dates


def get_tickers_for_prefetch(backtest_start: str, backtest_end: str, years_back: int = 4) -> List[str]:
    """
    Gets tickers that should be included in prefetch for a backtesting period.
    Only includes tickers that were listed on the stock exchange during the backtesting period.
    
    Args:
        backtest_start (str): Start date of backtesting period (YYYY-MM-DD)
        backtest_end (str): End date of backtesting period (YYYY-MM-DD)
        years_back (int): Number of years to look back for S&P 500 historical data (default: 4)
    
    Returns:
        List[str]: Tickers that were listed on exchange during backtesting period
    """
    # Get all filtered S&P 500 tickers from last N years (removes problematic currencies/delisted)
    filtered_tickers = get_sp500_tickers_filtered(years_back)
    if not filtered_tickers:
        logger.warning("No filtered S&P 500 tickers available")
        return []
    
    # Get IPO dates for all tickers
    ipo_dates = get_exchange_listing_dates(filtered_tickers)
    
    # Filter: only include tickers that were listed on exchange during backtesting period
    backtest_end_dt = datetime.strptime(backtest_end, '%Y-%m-%d')
    
    eligible_tickers = []
    excluded_count = 0
    
    for ticker in filtered_tickers:
        if ticker in ipo_dates:
            ipo_date = ipo_dates[ticker]
            # Normalize datetime objects for comparison (remove timezone info if present)
            if ipo_date.tzinfo is not None:
                ipo_date = ipo_date.replace(tzinfo=None)
            # Include if IPO was before or during backtesting period
            if ipo_date <= backtest_end_dt:
                eligible_tickers.append(ticker)
            else:
                excluded_count += 1
                logger.debug(f"Excluding {ticker}: IPO {ipo_date.strftime('%Y-%m-%d')} after backtest end {backtest_end}")
        else:
            # If no IPO date available, include to be safe (probably old ticker)
            eligible_tickers.append(ticker)
    
    logger.info(f"Prefetching data for {len(eligible_tickers)} tickers", module="tickers")
    
    return sorted(eligible_tickers)


def get_sp500_tickers_with_dates_filtered(years_back: int = 4) -> Dict[str, Dict]:
    """
    Gets S&P 500 tickers with their listing date information, filtered to remove problematic tickers.
    Combines the filtered ticker list with S&P 500 membership dates.
    
    Args:
        years_back (int): Number of years to look back (default: 4).
    
    Returns:
        Dict[str, Dict]: {ticker: {'sp500_start': datetime, 'sp500_end': datetime|None}}
                        Only includes filtered (valid) tickers.
    """
    # Get filtered ticker list (removes problematic tickers like currencies, truly delisted companies)
    filtered_tickers = get_sp500_tickers_filtered(years_back)
    if not filtered_tickers:
        logger.warning("No filtered tickers available")
        return {}
    
    # Get ticker membership data (includes all tickers with dates)
    all_ticker_dates = get_sp500_tickers_with_dates(years_back)
    if not all_ticker_dates:
        logger.warning("No ticker membership data available")
        return {}
    
    # Return intersection: only filtered tickers with their S&P 500 membership dates
    filtered_ticker_dates = {
        ticker: dates for ticker, dates in all_ticker_dates.items() 
        if ticker in filtered_tickers
    }
    
    logger.debug(f"Filtered ticker dates: {len(filtered_ticker_dates)} valid tickers "
                f"(removed {len(all_ticker_dates) - len(filtered_ticker_dates)} problematic)", 
                module="get_sp500_tickers_with_dates_filtered")
    
    return filtered_ticker_dates


def get_eligible_tickers_for_date(target_date: str, min_days_listed: int = 200, years_back: int = 4) -> List[str]:
    """
    Gets tickers that were eligible for analysis on a specific date.
    A ticker is eligible if it has been listed in the S&P 500 for at least min_days_listed days.
    
    Args:
        target_date (str): Date to check eligibility for (YYYY-MM-DD format).
        min_days_listed (int): Minimum days a ticker must be listed (default: 200) becasue of the lookback period.
        years_back (int): Number of years to look back for historical data (default: 4).
    
    Returns:
        List[str]: List of eligible ticker symbols.
    """
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    
    # Get filtered S&P 500 tickers and their membership data (excludes problematic tickers)
    ticker_dates = get_sp500_tickers_with_dates_filtered(years_back)
    filtered_tickers = list(ticker_dates.keys())
    
    # Get IPO dates for all filtered tickers
    ipo_dates = get_exchange_listing_dates(filtered_tickers)
    
    eligible_tickers = []
    
    for ticker, dates in ticker_dates.items():
        sp500_start = dates['sp500_start']
        sp500_end = dates['sp500_end']
        
        # Check if ticker was in S&P 500 on target date
        if sp500_start <= target_dt and (sp500_end is None or sp500_end >= target_dt):
            # Check if ticker has been listed on exchange for at least min_days_listed
            if ticker in ipo_dates:
                ipo_date = ipo_dates[ticker]
                days_listed = (target_dt - ipo_date).days
                if days_listed >= min_days_listed:
                    eligible_tickers.append(ticker)
                else:
                    logger.debug(f"Excluding {ticker}: only {days_listed} days listed on exchange (need {min_days_listed})")
            else:
                # If no IPO date available, fall back to S&P 500 start date
                days_listed = (target_dt - sp500_start).days
                if days_listed >= min_days_listed:
                    eligible_tickers.append(ticker)
                    logger.debug(f"Using S&P 500 start date for {ticker} (no IPO date available)")
    
    logger.debug(f"Found {len(eligible_tickers)} eligible tickers for {target_date} "
                f"(min {min_days_listed} days listed)", module="get_eligible_tickers_for_date")
    
    return sorted(eligible_tickers)


def normalize_ticker(ticker: str) -> str:
    """
    Normalizes ticker symbols for compatibility with yfinance.
    
    Args:
        ticker (str): Original ticker symbol
        
    Returns:
        str: Normalized ticker symbol
    """
    # Handle Class B shares - replace dot with dash
    if '.' in ticker:
        ticker = ticker.replace('.', '-')
    
    return ticker.upper()


def get_sp500_tickers_wikipedia() -> List[str]:
    """
    Scrapes S&P 500 tickers from Wikipedia and normalizes them.
    
    Returns:
        list: S&P 500 tickers normalized for yfinance compatibility.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        tables = pd.read_html(url)
        sp500_table = tables[0]
        
        tickers = sp500_table['Symbol'].tolist()
        # Normalize tickers for yfinance compatibility
        tickers = [normalize_ticker(ticker) for ticker in tickers]
        return tickers
        
    except Exception as e:
        logger.warning(f"Error fetching S&P 500 tickers: {e}")
        return []


def get_problematic_tickers(tickers: List[str]) -> List[str]:
    """
    Function to determine which tickers have no data available in yfinance.
    Excludes reassigned tickers and identifies delisted/problematic securities.
    
    Args:
        tickers: List of ticker symbols to test
        
    Returns:
        List of tickers that have no data (truly delisted/problematic)
    """
    import yfinance as yf
    
    # Known problematic tickers to exclude
    excluded_tickers = {
        'FB',    # Facebook -> META (FB is now ProShares ETF)
        'EGG',   # Cal-Maine Foods -> Enigmatig Limited
        'AUD',   # Australian Dollar currency 
        'COG',   # Cabot Oil & Gas (acquired and delisted)
        'FRC',   # First Republic Bank (failed and acquired by JPMorgan)
        'NLSN',  # Nielsen (acquired and went private)
        'TWTR',  # Twitter (went private)
        'X',     # United States Steel Corporation (delisted June 30, 2025)
        'INFO',  # IHS Markit (delisted and no more data available)
        'SPC',   # CrossingBridge Pre-Merger SPAC ETF (ETF, not stock)
    }
    
    problematic = []
    total = len(tickers)
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:3d}/{total}] Testing {ticker}...", end=" ", flush=True)
        
        # Skip known problematic tickers
        if ticker in excluded_tickers:
            problematic.append(ticker)
            print("EXCLUDED (known problematic)")
            continue
        
        try:
            stock = yf.Ticker(ticker)
            
            # Try to get recent price data with multiple approaches
            recent_data_found = False
            
            # Test 1: Try 1 month period
            try:
                recent = stock.history(period='1mo')
                if not recent.empty and len(recent) > 1:  # More than 1 day of data
                    recent_data_found = True
            except Exception as e:
                if "Period" in str(e) and "invalid" in str(e):
                    # This ticker doesn't support period-based queries
                    problematic.append(ticker)
                    print("PROBLEMATIC (period error)")
                    continue
            
            # Test 2: If period failed, try date range
            if not recent_data_found:
                try:
                    from datetime import datetime, timedelta
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30)
                    recent = stock.history(start=start_date.strftime('%Y-%m-%d'), 
                                         end=end_date.strftime('%Y-%m-%d'))
                    if not recent.empty and len(recent) > 1:
                        recent_data_found = True
                except Exception:
                    pass
            
            # Test 3: Check if it's a currency or forex ticker
            try:
                info = stock.info
                if info:
                    quote_type = info.get('quoteType', '')
                    long_name = info.get('longName', '').lower()
                    
                    # Check for currency indicators
                    if (quote_type in ['CURRENCY', 'FOREX'] or 
                        'currency' in long_name or 
                        ticker.upper() in ['AUD', 'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'CHF', 'CNY']):
                        problematic.append(ticker)
                        print("CURRENCY (not a stock)")
                        continue
            except Exception:
                pass
            
            if recent_data_found:
                print("OK")
            else:
                problematic.append(ticker)
                print("PROBLEMATIC (no recent data)")
                
        except Exception as e:
            problematic.append(ticker)
            # Provide more specific error information
            error_str = str(e)
            if "Period" in error_str and "invalid" in error_str:
                print("PROBLEMATIC (period error)")
            elif "404" in error_str or "not found" in error_str.lower():
                print("PROBLEMATIC (not found)")
            else:
                print(f"ERROR ({error_str[:30]}...)")
    
    return problematic


def get_sp500_delisted_tickers(years_back: int = 4) -> List[str]:
    """
    Gets S&P 500 tickers that were delisted (removed from index) in the last specified years.
    Returns tickers that were in the historical S&P 500 but are NOT in the current index.
    
    Args:
        years_back (int): Number of years to look back (default: 4).
    
    Returns:
        List[str]: Tickers that were delisted from S&P 500 during the period.
    """
    # Get all historical S&P 500 tickers from the last N years
    historical_tickers = get_sp500_constituents_wrds(years_back)
    if not historical_tickers:
        logger.warning("Could not get historical S&P 500 tickers, falling back to empty list")
        return []
    
    # Get current S&P 500 tickers
    current_tickers = get_sp500_tickers_wikipedia()
    if not current_tickers:
        logger.warning("Could not get current S&P 500 tickers")
        return []
    
    # Find tickers that were in historical but not in current
    historical_set = set(historical_tickers)
    current_set = set(current_tickers)
    delisted_tickers = sorted(list(historical_set - current_set))
    
    logger.info(f"Found {len(delisted_tickers)} delisted S&P 500 tickers from last {years_back} years")
    logger.info(f"Historical tickers: {len(historical_tickers)}, Current tickers: {len(current_tickers)}")
    
    return delisted_tickers


def get_sp500_tickers_filtered(years_back: int = 4) -> List[str]:
    """
    Gets historical S&P 500 tickers with problematic tickers removed.
    Uses caching to avoid re-testing tickers with yfinance repeatedly.
    
    Args:
        years_back (int): Number of years to look back (default: 4).
        
    Returns:
        List[str]: Historical S&P 500 tickers minus problematic ones
    """
    # Initialize cache
    cache = DiskCache()
    
    # Add filtered tickers cache type with 60 day refresh  
    if "filtered_tickers" not in cache.ttls:
        from diskcache import Cache
        cache.ttls["filtered_tickers"] = timedelta(days=60)  # 60 day refresh
        cache.caches["filtered_tickers"] = Cache(CACHE_DIR / "filtered_tickers")
    
    # Reuse existing cache instance if already created
    elif "filtered_tickers" not in cache.caches:
        from diskcache import Cache
        cache.caches["filtered_tickers"] = Cache(CACHE_DIR / "filtered_tickers")
    
    # Check cache first
    cache_key = f"sp500_filtered_tickers:{years_back}y"
    cached_filtered = cache.caches["filtered_tickers"].get(cache_key)
    
    if cached_filtered:
        return cached_filtered
    
    # Cache miss - need to fetch and filter
    logger.info("Cached filtered tickers not found. Fetching fresh data...")
    
    # Get all historical tickers (always fresh from WRDS)
    historical_tickers = get_sp500_constituents_wrds(years_back)
    if not historical_tickers:
        return []
    
    # Get current tickers to identify which ones to test
    current_tickers = get_sp500_tickers_wikipedia()
    current_set = set(current_tickers)
    
    # Only test delisted tickers (not in current S&P 500)
    delisted_tickers = [t for t in historical_tickers if t not in current_set]
    
    logger.info(f"Historical tickers: {len(historical_tickers)}")
    logger.info(f"Current tickers: {len(current_tickers)}")
    logger.info(f"Delisted tickers to test: {len(delisted_tickers)}")
    
    # Find problematic tickers among delisted ones only
    logger.info(f"Testing {len(delisted_tickers)} delisted tickers with yfinance...")
    problematic = get_problematic_tickers(delisted_tickers)
    
    # Remove problematic tickers from historical list
    filtered_tickers = [t for t in historical_tickers if t not in problematic]
    
    # Cache the filtered results
    cache.caches["filtered_tickers"].set(
        cache_key, 
        filtered_tickers, 
        expire=cache._get_expiration("filtered_tickers")
    )
    
    logger.info(f"Historical tickers: {len(historical_tickers)}")
    logger.info(f"Problematic tickers: {len(problematic)}")
    logger.info(f"Filtered tickers: {len(filtered_tickers)}")
    logger.info(f"Cached filtered results for future use")
    
    return filtered_tickers


def get_sp500_tickers(use_wrds: bool = True, years_back: int = 4, filter_problematic: bool = True) -> List[str]:
    """
    Gets S&P 500 tickers with fallback mechanism.
    
    Args:
        use_wrds (bool): Whether to try WRDS first (default: True).
        years_back (int): Number of years to look back for WRDS data (default: 4).
        filter_problematic (bool): Whether to filter out problematic tickers (default: True).
    
    Returns:
        list: S&P 500 tickers (filtered historical tickers if WRDS, current if Wikipedia).
    """
    if use_wrds and wrds is not None:
        # Try WRDS first - get filtered historical tickers by default
        if filter_problematic:
            return get_sp500_tickers_filtered(years_back)
        else:
            return get_sp500_constituents_wrds(years_back)
    
    # Fallback to Wikipedia
    return get_sp500_tickers_wikipedia()

# Example usage
if __name__ == "__main__":
    #nyse_tickers = get_nyse_tickers(include_etfs=False)
    #logger.info(f"First 10 NYSE tickers: {nyse_tickers[:10]}")
    
    sp500_tickers = get_sp500_tickers()
    logger.info(f"{len(sp500_tickers)} S&P 500 tickers: {sp500_tickers}")
    
    # Download and cache IPO dates for all filtered S&P 500 tickers
    ipo_dates = get_exchange_listing_dates(sp500_tickers)
    logger.info(f"Downloaded IPO dates for {len(ipo_dates)} tickers")