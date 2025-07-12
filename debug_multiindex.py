#!/usr/bin/env python3
"""
Debug script to check which tickers return MultiIndex DataFrames from yfinance.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import yfinance as yf
import pandas as pd
from utils.tickers import get_eligible_tickers_for_date

def check_multiindex_issue():
    """Check if MultiIndex DataFrames are returned by yfinance for different tickers"""
    
    print("=== CHECKING MULTIINDEX DATAFRAMES FROM YFINANCE ===\n")
    
    # Get a sample of eligible tickers
    target_date = "2022-01-01"
    eligible_tickers = get_eligible_tickers_for_date(target_date, min_days_listed=200)
    
    # Test with BAC and a sample of other tickers
    test_tickers = ['BAC']  # Start with BAC (known problematic)
    
    # Add first 10 eligible tickers for comparison
    other_tickers = [t for t in eligible_tickers[:10] if t != 'BAC']
    test_tickers.extend(other_tickers)
    
    start_date = "2021-12-30"
    end_date = "2022-01-03"
    
    multiindex_tickers = []
    regular_tickers = []
    no_data_tickers = []
    
    for ticker in test_tickers:
        print(f"Testing {ticker}...")
        
        try:
            # Download data
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                print(f"  ❌ No data returned")
                no_data_tickers.append(ticker)
            else:
                print(f"  📊 Data shape: {data.shape}")
                print(f"  📋 Columns: {data.columns}")
                
                # Check if MultiIndex
                if isinstance(data.columns, pd.MultiIndex):
                    print(f"  🔴 MultiIndex detected!")
                    print(f"     Levels: {data.columns.levels}")
                    print(f"     Names: {data.columns.names}")
                    multiindex_tickers.append(ticker)
                    
                    # Show how to access volume correctly
                    if 'Volume' in data.columns.get_level_values(0):
                        volume_data = data['Volume'][ticker] if ticker in data.columns.get_level_values(1) else data['Volume']
                        print(f"     Volume access: data['Volume']['{ticker}'] or data['Volume']")
                        print(f"     Sample volume: {volume_data.iloc[0] if len(volume_data) > 0 else 'N/A'}")
                else:
                    print(f"  ✅ Regular DataFrame (not MultiIndex)")
                    regular_tickers.append(ticker)
                    
                    # Check volume data for regular DataFrames
                    if 'Volume' in data.columns:
                        volume_data = data['Volume']
                        print(f"     Volume access: data['Volume']")
                        print(f"     Sample volume: {volume_data.iloc[0] if len(volume_data) > 0 else 'N/A'}")
                
        except Exception as e:
            print(f"  💥 Error: {e}")
        
        print()
    
    # Summary
    print("=== SUMMARY ===")
    print(f"MultiIndex DataFrames: {len(multiindex_tickers)} tickers")
    if multiindex_tickers:
        print(f"  - {multiindex_tickers}")
    
    print(f"Regular DataFrames: {len(regular_tickers)} tickers") 
    if regular_tickers:
        print(f"  - {regular_tickers}")
    
    print(f"No data: {len(no_data_tickers)} tickers")
    if no_data_tickers:
        print(f"  - {no_data_tickers}")
    
    # Analysis
    print("\n=== ANALYSIS ===")
    if len(multiindex_tickers) == 1 and 'BAC' in multiindex_tickers:
        print("🎯 Only BAC returns MultiIndex - this is a BAC-specific issue")
    elif len(multiindex_tickers) > 1:
        print("⚠️  Multiple tickers return MultiIndex - this is a yfinance behavior")
    elif len(multiindex_tickers) == 0:
        print("✅ No tickers return MultiIndex - might be a different issue")
    else:
        print("🤔 Mixed results - needs further investigation")

def test_batch_vs_individual():
    """Test if batch download vs individual download affects MultiIndex behavior"""
    
    print("\n=== TESTING BATCH VS INDIVIDUAL DOWNLOAD ===\n")
    
    test_tickers = ['BAC', 'AAPL', 'MSFT']
    start_date = "2021-12-30"
    end_date = "2022-01-03"
    
    print("1. Individual downloads:")
    for ticker in test_tickers:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        is_multiindex = isinstance(data.columns, pd.MultiIndex)
        print(f"   {ticker}: {'MultiIndex' if is_multiindex else 'Regular'}")
    
    print("\n2. Batch download:")
    batch_data = yf.download(test_tickers, start=start_date, end=end_date, progress=False)
    is_multiindex = isinstance(batch_data.columns, pd.MultiIndex)
    print(f"   Batch: {'MultiIndex' if is_multiindex else 'Regular'}")
    
    if is_multiindex:
        print(f"   Batch levels: {batch_data.columns.levels}")

if __name__ == "__main__":
    check_multiindex_issue()
    test_batch_vs_individual()