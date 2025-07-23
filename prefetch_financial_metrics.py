#!/usr/bin/env python3
"""
Script to prefetch financial metrics for all S&P 500 tickers.
This populates the monthly cache to make hedge fund runs faster.

The financial metrics cache uses monthly cache periods (YYYY-MM format) 
based on the end_date parameter. This script generates cache entries 
for multiple months to ensure comprehensive coverage.

Usage:
    python prefetch_financial_metrics.py              # Prefetch for last 4 years
    python prefetch_financial_metrics.py --years 2    # Prefetch for last 2 years
    python prefetch_financial_metrics.py --start AAPL # Resume from specific ticker
"""

import os
import sys
import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import time

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.tickers import get_sp500_tickers_filtered
from tools.api import fetch_financial_metrics_async, get_monthly_cache_period
from utils.logger import logger


def generate_monthly_periods(years_back: int = 4) -> List[str]:
    """
    Generate monthly cache periods for the specified number of years.
    
    Args:
        years_back: Number of years to generate periods for
        
    Returns:
        List of monthly periods in YYYY-MM format
    """
    periods = []
    current_date = datetime.now()
    
    # Generate periods for each month in the time range
    for year in range(current_date.year - years_back, current_date.year + 1):
        start_month = 1 if year != current_date.year else 1
        end_month = 12 if year != current_date.year else current_date.month
        
        for month in range(start_month, end_month + 1):
            periods.append(f"{year}-{month:02d}")
    
    return periods


def get_end_date_for_period(period: str) -> str:
    """
    Convert a monthly period to an end_date suitable for cache key generation.
    
    Args:
        period: Monthly period in YYYY-MM format
        
    Returns:
        End date in YYYY-MM-DD format (last day of the month)
    """
    year, month = map(int, period.split('-'))
    
    # Get the last day of the month
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    
    last_day = next_month - timedelta(days=1)
    return last_day.strftime('%Y-%m-%d')


async def prefetch_financial_metrics(years_back: int = 4, start_ticker: str = None):
    """
    Prefetch financial metrics for all S&P 500 tickers across multiple time periods.
    
    Args:
        years_back: Number of years of data to prefetch
        start_ticker: Ticker to start from (useful for resuming)
    """
    logger.info(f"Starting financial metrics prefetch for {years_back} years")
    
    # Get all filtered tickers
    all_tickers = get_sp500_tickers_filtered()
    logger.info(f"Found {len(all_tickers)} filtered S&P 500 tickers")
    
    # Generate monthly periods to cache
    periods = generate_monthly_periods(years_back)
    logger.info(f"Generated {len(periods)} monthly periods: {periods[0]} to {periods[-1]}")
    
    # Handle start_ticker parameter for resuming
    start_index = 0
    if start_ticker:
        if start_ticker in all_tickers:
            start_index = all_tickers.index(start_ticker)
            logger.info(f"Resuming from ticker {start_ticker} (index {start_index})")
        else:
            logger.warning(f"Start ticker {start_ticker} not found in ticker list. Starting from beginning.")
    
    # Process each monthly period
    total_periods = len(periods)
    for period_idx, period in enumerate(periods):
        logger.info(f"Processing period {period} ({period_idx + 1}/{total_periods})")
        
        # Convert period to end_date for API call
        end_date = get_end_date_for_period(period)
        
        # Verify this generates the correct cache key
        expected_cache_period = get_monthly_cache_period(end_date)
        if expected_cache_period != period:
            logger.warning(f"Cache period mismatch: expected {period}, got {expected_cache_period}")
            continue
        
        # Process tickers in batches to avoid overwhelming the system
        batch_size = 50  # Reasonable batch size for parallel processing
        tickers_to_process = all_tickers[start_index:] if period_idx == 0 else all_tickers
        
        for batch_start in range(0, len(tickers_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(tickers_to_process))
            ticker_batch = tickers_to_process[batch_start:batch_end]
            
            logger.info(f"  Processing batch {batch_start//batch_size + 1} "
                       f"({batch_start + 1}-{batch_end}/{len(tickers_to_process)}): {len(ticker_batch)} tickers")
            
            try:
                # Fetch financial metrics for this batch and period
                # This will automatically cache the results using the monthly period
                await fetch_financial_metrics_async(
                    tickers=ticker_batch,
                    end_date=end_date,
                    period="annual",  # Use annual data
                    limit=10,
                    verbose_data=False
                )
                
                logger.info(f"  ✓ Completed batch {batch_start//batch_size + 1} for period {period}")
                
                # Small delay between batches to be respectful to APIs
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"  ✗ Error processing batch {batch_start//batch_size + 1} for period {period}: {e}")
                # Continue with next batch rather than failing entirely
                continue
        
        # Delay between periods to avoid rate limiting
        if period_idx < total_periods - 1:
            logger.info(f"Completed period {period}, waiting before next period...")
            await asyncio.sleep(2)
    
    logger.info("✅ Financial metrics prefetch completed!")
    logger.info(f"📊 Summary:")
    logger.info(f"   • Processed {len(periods)} monthly periods")
    logger.info(f"   • Processed {len(all_tickers)} tickers")
    logger.info(f"   • Total cache entries: {len(periods) * len(all_tickers)} (theoretical max)")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Prefetch financial metrics for S&P 500 tickers')
    parser.add_argument('--years', type=int, default=4, 
                       help='Number of years of data to prefetch (default: 4)')
    parser.add_argument('--start', type=str, default=None,
                       help='Ticker symbol to start from (for resuming)')
    
    args = parser.parse_args()
    
    # Run the async prefetch function
    asyncio.run(prefetch_financial_metrics(args.years, args.start))


if __name__ == "__main__":
    main()