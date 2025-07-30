#!/usr/bin/env python3
"""
Debug script to investigate price data retrieval issues.
This script will systematically test the price data pipeline to identify
where and why the data retrieval is failing.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
from datetime import datetime, timedelta
from src.tools.api import get_price_data, get_prices, _cache
from src.utils.market_calendar import is_trading_day, get_previous_trading_day, get_next_trading_day
import yfinance as yf

def test_problematic_dates_and_tickers():
    """Test the specific date range and tickers that are causing issues"""
    print("=== TESTING PROBLEMATIC DATES AND TICKERS ===\n")
    
    # Test dates from the error log
    test_cases = [
        {
            'ticker': 'SHY',
            'start_date': '2024-01-02', 
            'end_date': '2024-01-03',
            'description': 'SHY ETF - New Year period'
        },
        {
            'ticker': 'AAPL',
            'start_date': '2024-01-02', 
            'end_date': '2024-01-03',
            'description': 'AAPL - Same period for comparison'
        },
        {
            'ticker': 'SPY',
            'start_date': '2024-01-02', 
            'end_date': '2024-01-03',
            'description': 'SPY ETF - Same period for comparison'
        },
        {
            'ticker': 'SHY',
            'start_date': '2023-12-29', 
            'end_date': '2024-01-02',
            'description': 'SHY ETF - Extended period from previous error'
        }
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"Test {i}: {case['description']}")
        print(f"Ticker: {case['ticker']}, Date range: {case['start_date']} to {case['end_date']}")
        
        # Check if dates are trading days
        start_trading = is_trading_day(case['start_date'])
        end_trading = is_trading_day(case['end_date'])
        print(f"Trading days - Start: {start_trading}, End: {end_trading}")
        
        # Test our API function
        try:
            df = get_price_data(case['ticker'], case['start_date'], case['end_date'], verbose_data=True)
            print(f"API Result - Shape: {df.shape}, Empty: {df.empty}")
            if not df.empty:
                print(f"Date range in data: {df.index.min()} to {df.index.max()}")
                print(f"Sample data:\n{df.head()}")
            else:
                print("DataFrame is EMPTY!")
        except Exception as e:
            print(f"API Error: {e}")
            df = pd.DataFrame()
        
        # Test direct yfinance access
        try:
            yf_ticker = yf.Ticker(case['ticker'])
            # Add one day to end_date for yfinance (it uses exclusive end)
            end_date_yf = (datetime.strptime(case['end_date'], '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            yf_data = yf_ticker.history(start=case['start_date'], end=end_date_yf)
            print(f"Direct yfinance - Shape: {yf_data.shape}, Empty: {yf_data.empty}")
            if not yf_data.empty:
                print(f"yfinance columns: {list(yf_data.columns)}")
                print(f"yfinance date range: {yf_data.index.min()} to {yf_data.index.max()}")
        except Exception as e:
            print(f"yfinance Error: {e}")
            yf_data = pd.DataFrame()
        
        results.append({
            'case': case,
            'api_empty': df.empty,
            'api_shape': df.shape,
            'yf_empty': yf_data.empty if 'yf_data' in locals() else True,
            'yf_shape': yf_data.shape if 'yf_data' in locals() else (0, 0)
        })
        
        print("-" * 80)
    
    return results

def test_cache_structure():
    """Investigate the cache structure and data format"""
    print("\n=== TESTING CACHE STRUCTURE ===\n")
    
    # Test specific ticker that's failing
    ticker = 'SHY'
    start_date = '2024-01-02'
    end_date = '2024-01-03'
    
    print(f"Testing cache for {ticker}")
    
    # Check what's in the cache
    cached_data = _cache.get_prices(ticker, start_date, end_date)
    print(f"Cached data type: {type(cached_data)}")
    print(f"Cached data: {cached_data}")
    
    # Check raw cache (no date filtering)
    raw_cached_data = _cache.get_prices(ticker)
    print(f"Raw cached data type: {type(raw_cached_data)}")
    if raw_cached_data:
        print(f"Raw cached data length: {len(raw_cached_data)}")
        if len(raw_cached_data) > 0:
            print(f"First cached item: {raw_cached_data[0]}")
            print(f"Sample cached items (first 3):")
            for item in raw_cached_data[:3]:
                print(f"  {item}")
        
        # Check date range in cache
        if raw_cached_data:
            dates = [item.get('time', 'NO_TIME') for item in raw_cached_data]
            dates = [d for d in dates if d != 'NO_TIME']
            if dates:
                print(f"Cached date range: {min(dates)} to {max(dates)}")
    else:
        print("No raw cached data found")

def test_prices_to_df_conversion():
    """Test the conversion from Price objects to DataFrame"""
    print("\n=== TESTING PRICES TO DATAFRAME CONVERSION ===\n")
    
    from src.tools.api import prices_to_df, get_prices
    
    ticker = 'SHY'
    start_date = '2024-01-02'
    end_date = '2024-01-03'
    
    print(f"Testing price conversion for {ticker}")
    
    # Get raw prices
    prices = get_prices(ticker, start_date, end_date, verbose_data=True)
    print(f"Raw prices type: {type(prices)}")
    print(f"Raw prices length: {len(prices)}")
    print(f"Raw prices: {prices}")
    
    # Convert to DataFrame
    df = prices_to_df(prices)
    print(f"Converted DataFrame shape: {df.shape}")
    print(f"Converted DataFrame empty: {df.empty}")
    print(f"Converted DataFrame:\n{df}")

def test_direct_yfinance_batch():
    """Test yfinance batch download directly to see what's happening"""
    print("\n=== TESTING DIRECT YFINANCE BATCH DOWNLOAD ===\n")
    
    tickers = ['SHY', 'AAPL', 'SPY']
    start_date = '2024-01-02'
    end_date = '2024-01-04'  # yfinance uses exclusive end
    
    print(f"Testing yfinance batch download for {tickers}")
    print(f"Date range: {start_date} to {end_date}")
    
    try:
        data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date, 
            group_by='ticker',
            auto_adjust=True,
            progress=True,
            timeout=20
        )
        
        print(f"Downloaded data type: {type(data)}")
        print(f"Downloaded data shape: {data.shape}")
        print(f"Downloaded data empty: {data.empty}")
        
        if isinstance(data.columns, pd.MultiIndex):
            print(f"MultiIndex levels: {data.columns.levels}")
            print(f"MultiIndex columns sample: {data.columns[:10]}")
            
            # Test access for SHY
            if 'SHY' in data.columns.levels[0]:
                shy_data = data['SHY']
                print(f"SHY data shape: {shy_data.shape}")
                print(f"SHY data:\n{shy_data}")
            else:
                print("SHY not found in level 0")
                
        print(f"Data index: {data.index}")
        print(f"Data sample:\n{data.head()}")
        
    except Exception as e:
        print(f"yfinance batch download error: {e}")
        import traceback
        traceback.print_exc()

def run_all_tests():
    """Run all diagnostic tests"""
    print("PRICE DATA RETRIEVAL DEBUG ANALYSIS")
    print("=" * 50)
    
    # Test 1: Problematic cases
    results = test_problematic_dates_and_tickers()
    
    # Test 2: Cache structure
    test_cache_structure()
    
    # Test 3: Conversion process
    test_prices_to_df_conversion()
    
    # Test 4: Direct yfinance
    test_direct_yfinance_batch()
    
    # Summary
    print("\n=== SUMMARY ===")
    print("Analysis complete. Key findings:")
    
    failed_cases = [r for r in results if r['api_empty']]
    if failed_cases:
        print(f"Failed API cases: {len(failed_cases)}")
        for case in failed_cases:
            print(f"  - {case['case']['ticker']}: {case['case']['start_date']} to {case['case']['end_date']}")
    else:
        print("All API cases succeeded")

if __name__ == "__main__":
    run_all_tests()