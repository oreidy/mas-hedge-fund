#!/usr/bin/env python3

import sys
import os

# Add src to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.logger import logger, LogLevel
from utils.tickers import get_tickers_for_prefetch, get_exchange_listing_date

def test_ipo_filtering():
    """Test IPO date filtering for problematic tickers"""
    
    logger.set_level('DEBUG')
    
    # Test specific problematic tickers
    problematic_tickers = ['AMTM', 'GEV', 'PHIN', 'VLTO', 'SPC']
    backtest_start = "2022-01-01"
    backtest_end = "2022-01-31"
    
    print(f"Testing IPO date filtering for problematic tickers")
    print(f"Backtest period: {backtest_start} to {backtest_end}")
    print()
    
    # Test individual IPO dates
    print("=== Individual IPO Dates ===")
    for ticker in problematic_tickers:
        ipo_date = get_exchange_listing_date(ticker)
        if ipo_date:
            ipo_str = ipo_date.strftime('%Y-%m-%d')
            should_exclude = ipo_date.strftime('%Y-%m-%d') > backtest_end
            print(f"{ticker}: IPO {ipo_str} -> {'EXCLUDE' if should_exclude else 'INCLUDE'}")
        else:
            print(f"{ticker}: No IPO date found")
    
    print()
    
    # Test prefetch filtering
    print("=== Prefetch Filtering Test ===")
    eligible_tickers = get_tickers_for_prefetch(backtest_start, backtest_end, years_back=4)
    
    print(f"Total eligible tickers for prefetch: {len(eligible_tickers)}")
    
    for ticker in problematic_tickers:
        if ticker in eligible_tickers:
            print(f"✅ {ticker}: INCLUDED in prefetch")
        else:
            print(f"❌ {ticker}: EXCLUDED from prefetch")

if __name__ == "__main__":
    test_ipo_filtering()