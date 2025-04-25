import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import pickle

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
                print(f"Loaded NYSE calendar from cache: {len(calendar_data)} trading days.")    
                return calendar_data
        except Exception as e:
            print(f"Error loading calendar cache: {e}")
    
    # If cache doesn't exist or is invalid, create new calendar
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"
    
    nyse = mcal.get_calendar('NYSE')
    calendar = nyse.valid_days(start_date=start, end_date=end)
    
    # Convert to set of dates for faster lookup
    calendar_dates = set(pd.to_datetime(calendar).date)
    print(f"Fetched NYSE calendar: {len(calendar_dates)} trading days from {start} to {end}.")
    
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
    print(f"Checking if {date} is a trading day: {is_trading}")
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
    If start_date is not a trading day, use the next trading day.
    If end_date is not a trading day, use the previous trading day.
    """
    print(f"Original date range: {start_date} to {end_date}")
    
    original_start = start_date
    original_end = end_date

    if not is_trading_day(start_date):
        start_date = get_next_trading_day(start_date)
        print(f"Adjusted start date from {original_start} to {start_date} (next trading day)")
    
    if not is_trading_day(end_date):
        end_date = get_previous_trading_day(end_date)
        print(f"Adjusted end date from {original_end} to {end_date} (previous trading day)")
    
    print(f"Final adjusted date range: {start_date} to {end_date}")
    
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
    print(f"NYSE Trading Calendar ({len(calendar_list)} days)")
    print(f"First 10 trading days: {calendar_list[:10]}")
    print(f"Last 10 trading days: {calendar_list[-10:]}")
    
    # Check specific ranges
    jan_2024_days = [d for d in calendar_list if d.year == 2024 and d.month == 1]
    print(f"\nJanuary 2024 trading days: {jan_2024_days}")
    
    # Check specific dates
    test_dates = ["2024-01-03", "2024-01-04", "2024-01-05"]
    for date_str in test_dates:
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        is_trading = date in calendar
        print(f"Is {date_str} a trading day? {is_trading}")
    
    return calendar