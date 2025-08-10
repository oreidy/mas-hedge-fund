#!/usr/bin/env python3
"""
Script to prefetch market cap data for all S&P 500 tickers.
This populates the cache with calculated market cap values to reduce API calls during backtesting.

The script leverages existing cached price and outstanding shares data to calculate market cap
values efficiently, falling back to API calls only when necessary.

Usage:
    python prefetch_market_cap.py              # Prefetch for last 4 years
    python prefetch_market_cap.py --years 2    # Prefetch for last 2 years
    python prefetch_market_cap.py --start AAPL # Resume from specific ticker
"""

import os
import sys
import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import time
import pandas as pd

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tickers import get_sp500_tickers_filtered
from tools.api import get_historical_market_cap
from utils.logger import logger
from utils.market_calendar import is_trading_day, get_previous_trading_day


def generate_trading_dates(years_back: int = 4) -> List[str]:
    """
    Generate ALL trading dates for the specified number of years.
    This gives us daily market cap coverage.
    
    Args:
        years_back: Number of years to generate dates for
        
    Returns:
        List of all trading dates in YYYY-MM-DD format
    """
    dates = []
    current_date = datetime.now()
    start_date = current_date - timedelta(days=years_back * 365)
    
    # Generate all trading days in the range
    current = start_date
    while current <= current_date:
        date_str = current.strftime('%Y-%m-%d')
        if is_trading_day(date_str):
            dates.append(date_str)
        current += timedelta(days=1)
    
    return dates


def prefetch_market_cap_data(years_back: int = 4, start_ticker: str = None):
    """
    Prefetch market cap data for all S&P 500 tickers across multiple dates.
    
    Args:
        years_back: Number of years of data to prefetch
        start_ticker: Ticker to start from (useful for resuming)
    """
    logger.info(f"Starting market cap prefetch for {years_back} years")
    
    # Get all filtered tickers
    all_tickers = get_sp500_tickers_filtered()
    logger.info(f"Found {len(all_tickers)} filtered S&P 500 tickers")
    
    # Generate trading dates to process
    trading_dates = generate_trading_dates(years_back)
    logger.info(f"Generated {len(trading_dates)} trading dates: {trading_dates[0]} to {trading_dates[-1]}")
    
    # Handle start_ticker parameter for resuming
    start_index = 0
    if start_ticker:
        if start_ticker in all_tickers:
            start_index = all_tickers.index(start_ticker)
            logger.info(f"Resuming from ticker {start_ticker} (index {start_index})")
        else:
            logger.warning(f"Start ticker {start_ticker} not found in ticker list. Starting from beginning.")
    
    # Track statistics
    errors = 0
    
    # Process each ticker
    tickers_to_process = all_tickers[start_index:]
    for ticker_idx, ticker in enumerate(tickers_to_process):
        logger.info(f"Processing ticker {ticker} ({ticker_idx + 1 + start_index}/{len(all_tickers)})")
        
        ticker_errors = 0
        
        # Process each date for this ticker
        for date in trading_dates:
            try:
                # This leverages cached price data and outstanding shares
                # The get_historical_market_cap function is already optimized to:
                # 1. Check market cap cache first 
                # 2. Use cached price data if available
                # 3. Only make API calls as last resort
                market_cap = get_historical_market_cap(ticker, date, verbose_data=False)
                
                if market_cap is None:
                    ticker_errors += 1
                    
            except Exception as e:
                ticker_errors += 1
                logger.warning(f"Error calculating market cap for {ticker} at {date}: {e}")
                continue
        
        errors += ticker_errors
        
        # Progress update
        if ticker_errors > 0:
            logger.info(f"  ✓ Completed {ticker}: {len(trading_dates) - ticker_errors}/{len(trading_dates)} successful")
        else:
            logger.info(f"  ✓ Completed {ticker}: {len(trading_dates)}/{len(trading_dates)} successful")
        
        # Brief delay between tickers (most data should come from cache)
        time.sleep(0.1)
    
    logger.info("✅ Market cap prefetch completed!")
    logger.info(f"📊 Summary:")
    logger.info(f"   • Processed {len(tickers_to_process)} tickers")
    logger.info(f"   • Processed {len(trading_dates)} dates per ticker")
    logger.info(f"   • Total calculations attempted: {len(tickers_to_process) * len(trading_dates)}")
    logger.info(f"   • Successful calculations: {len(tickers_to_process) * len(trading_dates) - errors}")
    logger.info(f"   • Errors: {errors}")
    
    if errors > 0:
        error_rate = (errors / (len(tickers_to_process) * len(trading_dates))) * 100
        logger.info(f"   • Error rate: {error_rate:.2f}%")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Prefetch market cap data for S&P 500 tickers')
    parser.add_argument('--years', type=int, default=4, 
                       help='Number of years of data to prefetch (default: 4)')
    parser.add_argument('--start', type=str, default=None,
                       help='Ticker symbol to start from (for resuming)')
    
    args = parser.parse_args()
    
    # Run the prefetch function
    prefetch_market_cap_data(args.years, args.start)


if __name__ == "__main__":
    main()