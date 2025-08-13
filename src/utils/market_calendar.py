import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import pickle
from utils.logger import logger
from typing import Tuple, Union


@lru_cache(maxsize=1)
def get_nyse_calendar(start_year=2020, end_year=2025):
    """
    Get NYSE trading calendar with caching.
    This function is cached to avoid repeated API calls.
    """
    cache_file = Path('./cache/nyse_calendar.pkl')
    
    # Try to load from cache file
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                calendar_data = pickle.load(f)
                
            # Check if the calendar covers our date range
            if (min(calendar_data).year <= start_year and 
                max(calendar_data).year >= end_year):
                logger.debug(f"Loaded NYSE calendar from cache: {len(calendar_data)} trading days.", module="get_nyse_calendar")    
                return calendar_data
        except Exception as e:
            logger.error(f"Error loading calendar cache: {e}", module="get_nyse_calendar")
    
    # If cache doesn't exist or is invalid, create new calendar
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"
    
    nyse = mcal.get_calendar('NYSE')
    calendar = nyse.valid_days(start_date=start, end_date=end)
    
    # Convert to set of dates for faster lookup
    calendar_dates = set(pd.to_datetime(calendar).date)
    logger.debug(f"Fetched NYSE calendar: {len(calendar_dates)} trading days from {start} to {end}.", module="get_nyse_calendar")
    
    # Save to cache
    cache_file.parent.mkdir(exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(calendar_dates, f)
    
    return calendar_dates

def is_trading_day(date_str):
    """Check if a given date is a trading day"""
    if isinstance(date_str, str):
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
    elif isinstance(date_str, datetime):
        date = date_str.date()
    else:
        date = date_str
        
    calendar = get_nyse_calendar()
    is_trading = date in calendar
    logger.debug(f"Checking if {date} is a trading day: {is_trading}", module="is_trading_day")
    return is_trading

def get_previous_trading_day(date_str):
    """Get the previous trading day from a given date"""
    if isinstance(date_str, str):
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
    elif isinstance(date_str, datetime):
        date = date_str.date()    
    else:
        date = date_str
        
    calendar = get_nyse_calendar()
    
    current_date = date - timedelta(days=1)
    while current_date not in calendar:
        current_date -= timedelta(days=1)
        # Safety check to prevent infinite loops
        if (date - current_date).days > 30:
            raise ValueError(f"Could not find trading day within 30 days before {date}")
    
    return current_date.strftime('%Y-%m-%d')

def get_next_trading_day(date_str):
    """Get the next trading day from a given date"""
    if isinstance(date_str, str):
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
    elif isinstance(date_str, datetime):
        date = date_str.date()
    else:
        date = date_str
        
    calendar = get_nyse_calendar()
    
    current_date = date + timedelta(days=1)
    while current_date not in calendar:
        current_date += timedelta(days=1)
        # Safety check to prevent infinite loops
        if (current_date - date).days > 30:
            raise ValueError(f"Could not find trading day within 30 days after {date}")
    
    return current_date.strftime('%Y-%m-%d')

def adjust_date_range(start_date, end_date):
    """
    Adjust date range to valid trading days.
    If start_date is not a trading day, use the previous trading day.
    If end_date is not a trading day, use the previous trading day.
    """
    logger.debug(f"Original date range: {start_date} to {end_date}", module="adjust_date_range")
    
    original_start = start_date
    original_end = end_date

    if not is_trading_day(start_date):
        start_date = get_previous_trading_day(start_date)
        logger.debug(f"Adjusted start date from {original_start} to {start_date} (previous trading day)", module="adjust_date_range")
    
    if not is_trading_day(end_date):
        end_date = get_previous_trading_day(end_date)
        logger.debug(f"Adjusted end date from {original_end} to {end_date} (previous trading day)", module="adjust_date_range")
    
    return start_date, end_date

def check_nyse_calendar(start_year=2020, end_year=2025):
    """
    Check and print NYSE trading days for specific periods.
    Useful for debugging calendar issues.
    """
    calendar = get_nyse_calendar(start_year, end_year)
    
    # Convert to list and sort for easier viewing
    calendar_list = sorted(list(calendar))
    
    # Print the first and last few days
    logger.info(f"NYSE Trading Calendar ({len(calendar_list)} days)", module="check_nyse_calendar")
    logger.debug(f"First 10 trading days: {calendar_list[:10]}", module="check_nyse_calendar")
    logger.debug(f"Last 10 trading days: {calendar_list[-10:]}", module="check_nyse_calendar")
    
    # Check specific ranges
    jan_2024_days = [d for d in calendar_list if d.year == 2024 and d.month == 1]
    logger.debug(f"January 2024 trading days: {jan_2024_days}", module="check_nyse_calendar")
    
    # Check specific dates
    test_dates = ["2024-01-03", "2024-01-04", "2024-01-05"]
    for date_str in test_dates:
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        is_trading = date in calendar
        logger.debug(f"Is {date_str} a trading day? {is_trading}", module="check_nyse_calendar")
    
    return calendar

def adjust_yfinance_date_range(start_date: Union[str, datetime], 
                              end_date: Union[str, datetime]) -> Tuple[str, str]:
    """
    Adjust date range for yfinance API calls to handle the exclusive end date.
    
    Args:
        start_date: Start date as string (YYYY-MM-DD) or datetime object
        end_date: End date as string (YYYY-MM-DD) or datetime object
        
    Returns:
        Tuple of (start_date_str, end_date_str) adjusted for yfinance
    """
    # Convert to datetime if string
    if isinstance(start_date, str):
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start_dt = start_date
        
    if isinstance(end_date, str):
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_dt = end_date
    
    # Add one day to end_date to make it inclusive
    end_dt = end_dt + timedelta(days=1)
    
    # Convert back to strings
    start_date_str = start_dt.strftime('%Y-%m-%d')
    end_date_str = end_dt.strftime('%Y-%m-%d')
    
    return start_date_str, end_date_str