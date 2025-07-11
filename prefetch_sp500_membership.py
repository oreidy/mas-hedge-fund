#!/usr/bin/env python3
"""
Script to prefetch and cache S&P 500 membership data from WRDS.
This populates the local cache so backtesting runs faster.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.tickers import get_sp500_tickers_with_dates
from utils.logger import logger


def main():
    print("Prefetching S&P 500 membership data from WRDS...")
    
    # Fetch and cache membership data for the last 4 years
    ticker_dates = get_sp500_tickers_with_dates(years_back=4)
    
    if ticker_dates:
        print(f"✅ Successfully cached S&P 500 membership data for {len(ticker_dates)} tickers")
        
        # Show some sample data
        print("\nSample membership data:")
        for i, (ticker, dates) in enumerate(list(ticker_dates.items())[:5]):
            start_date = dates['sp500_start']
            end_date = dates['sp500_end'] if dates['sp500_end'] else 'Current'
            # Handle both datetime objects and strings
            if hasattr(start_date, 'strftime'):
                start_date = start_date.strftime('%Y-%m-%d')
            if end_date != 'Current' and hasattr(end_date, 'strftime'):
                end_date = end_date.strftime('%Y-%m-%d')
            print(f"  {ticker}: {start_date} to {end_date}")
        
        if len(ticker_dates) > 5:
            print(f"  ... and {len(ticker_dates) - 5} more tickers")
            
    else:
        print("❌ Failed to fetch S&P 500 membership data")
        print("Please check your WRDS credentials and connection")
        return 1
    
    print("\n🎉 S&P 500 membership data is now cached for fast backtesting!")
    return 0


if __name__ == "__main__":
    sys.exit(main())