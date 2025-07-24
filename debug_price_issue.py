#!/usr/bin/env python3
"""
Debug script to reproduce the missing price data issue for 2022-04-08
"""
import sys
import os
sys.path.append('src')

from tools.api import get_price_data, get_data_for_tickers, _cache
from utils.logger import logger
import pandas as pd

def test_individual_price_fetch():
    """Test individual price fetching like the backtester does"""
    print("=== Test 1: Individual Price Fetching ===")
    
    ticker = 'AAPL'
    previous_date = '2022-04-07'
    current_date = '2022-04-08'
    
    # Clear cache to simulate fresh run
    _cache.clear('prices')
    
    # This mimics what backtester does in run_backtest
    price_df = get_price_data(ticker, previous_date, current_date, verbose_data=False)
    print(f"Price data for {ticker} from {previous_date} to {current_date}:")
    print(f"  Shape: {price_df.shape}")
    print(f"  Dates: {list(price_df.index.strftime('%Y-%m-%d'))}")
    
    if len(price_df) < 2:
        print(f"❌ ISSUE REPRODUCED: Only {len(price_df)} rows returned, expected 2")
        
        # Test fallback method
        print("  Testing fallback individual queries...")
        prev_df = get_price_data(ticker, previous_date, previous_date, verbose_data=False)
        curr_df = get_price_data(ticker, current_date, current_date, verbose_data=False)
        
        fallback_dfs = [df for df in [prev_df, curr_df] if not df.empty]
        print(f"  Fallback results: {len(fallback_dfs)} non-empty DataFrames")
        
        if len(fallback_dfs) >= 1:
            combined_df = pd.concat(fallback_dfs).sort_index()
            print(f"  Combined fallback shape: {combined_df.shape}")
            print(f"  Combined dates: {list(combined_df.index.strftime('%Y-%m-%d'))}")
    else:
        print(f"✅ Success: Got {len(price_df)} rows as expected")


def test_batch_prefetch():
    """Test batch prefetching like the backtester does in prefetch_data"""
    print("\n=== Test 2: Batch Prefetching ===")
    
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2022-04-01'
    end_date = '2022-04-08'
    
    # Clear cache to simulate fresh run
    _cache.clear('prices')
    
    # This mimics what backtester does in prefetch_data
    data = get_data_for_tickers(tickers, start_date, end_date, verbose_data=False)
    
    print(f"Batch prefetch results:")
    for ticker in tickers:
        if ticker in data:
            prices_df = data[ticker]['prices']
            print(f"  {ticker}: {prices_df.shape[0]} price rows")
            if not prices_df.empty:
                date_range = f"{prices_df.index.min().strftime('%Y-%m-%d')} to {prices_df.index.max().strftime('%Y-%m-%d')}"
                print(f"    Date range: {date_range}")
                
                # Check if 2022-04-08 data is included
                if '2022-04-08' in prices_df.index.strftime('%Y-%m-%d'):
                    print(f"    ✅ 2022-04-08 data present")
                else:
                    print(f"    ❌ 2022-04-08 data missing")
            else:
                print(f"    ❌ Empty DataFrame")
        else:
            print(f"  {ticker}: Not in results")


def test_cache_behavior():
    """Test cache behavior to see if there are any issues"""
    print("\n=== Test 3: Cache Behavior ===")
    
    ticker = 'AAPL'
    start_date = '2022-04-07'
    end_date = '2022-04-08'
    
    # Clear cache
    _cache.clear('prices')
    print("Cache cleared")
    
    # First fetch - should populate cache
    print("First fetch (should populate cache):")
    df1 = get_price_data(ticker, start_date, end_date, verbose_data=False)
    print(f"  Returned {df1.shape[0]} rows")
    
    # Check what's in cache
    cached_data = _cache.get_prices(ticker)
    print(f"  Cache now contains {len(cached_data) if cached_data else 0} items")
    
    # Second fetch - should use cache
    print("Second fetch (should use cache):")
    df2 = get_price_data(ticker, start_date, end_date, verbose_data=False)
    print(f"  Returned {df2.shape[0]} rows")
    
    # Compare results
    if df1.shape == df2.shape and df1.equals(df2):
        print("  ✅ Cache behavior consistent")
    else:
        print("  ❌ Cache behavior inconsistent")
        print(f"  First fetch dates: {list(df1.index.strftime('%Y-%m-%d'))}")
        print(f"  Second fetch dates: {list(df2.index.strftime('%Y-%m-%d'))}")


if __name__ == '__main__':
    print("Debugging missing price data issue for 2022-04-08")
    print("=" * 60)
    
    test_individual_price_fetch()
    test_batch_prefetch()
    test_cache_behavior()
    
    print("\n" + "=" * 60)
    print("Debug script completed")