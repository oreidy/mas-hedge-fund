#!/usr/bin/env python3
"""
Quick fix script to prefetch only the bond ETF data needed by the backtester.
This will cache SHY and TLT data for the entire date range needed.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.disk_cache import DiskCache
from utils.logger import logger

def yfinance_to_cache_format(hist_df, ticker):
    """Convert yfinance DataFrame to cache format"""
    cache_data = []
    for date, row in hist_df.iterrows():
        try:
            cache_data.append({
                "time": date.strftime('%Y-%m-%d'),
                "open": float(row['Open']),
                "high": float(row['High']), 
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume']) if pd.notna(row['Volume']) else 0
            })
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping row for {ticker} on {date}: {e}")
            continue
    return cache_data

def prefetch_ticker_prices(ticker, start_date="2021-04-01", end_date="2025-07-01"):
    """Prefetch price data for a single ticker"""
    try:
        logger.info(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        
        # Get historical data - yfinance will automatically start from first available date
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            logger.warning(f"No data available for {ticker}")
            return False
        
        # Convert to cache format
        cache_data = yfinance_to_cache_format(hist, ticker)
        
        if not cache_data:
            logger.warning(f"No valid price data for {ticker} after conversion")
            return False
        
        # Save to cache
        cache = DiskCache()
        cache.set_prices(ticker, cache_data)
        
        date_range = f"{hist.index.min().strftime('%Y-%m-%d')} to {hist.index.max().strftime('%Y-%m-%d')}"
        logger.info(f"Cached {len(cache_data)} price records for {ticker} ({date_range})")
        return True
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return False

def main():
    """Main function to prefetch bond ETF prices only"""
    print("PREFETCHING BOND ETF DATA TO FIX BACKTESTER")
    print("=" * 50)
    
    # Bond ETFs used by the backtester
    bond_etfs = ["SHY", "TLT"]
    
    logger.info(f"Starting bond ETF prefetch for: {bond_etfs}")
    
    # Check current cache status first
    cache = DiskCache()
    for ticker in bond_etfs:
        cached_data = cache.get_prices(ticker)
        if cached_data:
            dates = [item['time'] for item in cached_data]
            dates.sort()
            print(f"{ticker} current cache: {len(cached_data)} records, {dates[0]} to {dates[-1]}")
            
            # Check for January 2024 data
            jan_2024_data = [d for d in dates if d.startswith('2024-01')]
            print(f"  - January 2024 data: {len(jan_2024_data)} days")
        else:
            print(f"{ticker} current cache: NO DATA")
    
    print("\nPrefetching bond ETF data...")
    
    # Track progress
    successful = 0
    failed = 0
    
    # Process each bond ETF
    for i, ticker in enumerate(bond_etfs, 1):
        print(f"\n[{i}/{len(bond_etfs)}] Processing {ticker} (Bond ETF)")
        
        if prefetch_ticker_prices(ticker):
            successful += 1
            
            # Verify the fix immediately
            cached_data = cache.get_prices(ticker)
            dates = [item['time'] for item in cached_data]
            dates.sort()
            
            # Check if January 2024 data is now available
            jan_2024_data = [d for d in dates if d.startswith('2024-01')]
            
            print(f"✅ {ticker} SUCCESS: {len(cached_data)} records, {dates[0]} to {dates[-1]}")
            print(f"   January 2024 data: {len(jan_2024_data)} days")
        else:
            failed += 1
            print(f"❌ {ticker} FAILED")
    
    # Summary
    print(f"\n" + "=" * 50)
    print(f"Bond ETF prefetch completed: {successful} successful, {failed} failed")
    
    # Test the fix with the problematic case
    print("\nTesting the fix with problematic date range...")
    
    from tools.api import get_price_data
    
    test_cases = [
        ("SHY", "2024-01-02", "2024-01-03"),
        ("TLT", "2024-01-02", "2024-01-03"),
        ("SHY", "2023-12-29", "2024-01-02"),
    ]
    
    for ticker, start, end in test_cases:
        try:
            df = get_price_data(ticker, start, end)
            status = "✅ SUCCESS" if not df.empty else "❌ STILL EMPTY"
            print(f"{ticker} {start} to {end}: {status} (shape: {df.shape})")
            
            if not df.empty:
                print(f"   Date range: {df.index.min()} to {df.index.max()}")
        except Exception as e:
            print(f"{ticker} {start} to {end}: ❌ ERROR - {e}")
    
    print(f"\n🎉 Fix complete! The backtester should now work without indexing errors.")

if __name__ == "__main__":
    main()