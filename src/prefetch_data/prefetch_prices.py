#!/usr/bin/env python3
"""
Prefetch historical price data for filtered S&P 500 tickers and populate the cache.
This script downloads price data from 2021-04-01 (or earliest available) to today's date.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tickers import get_sp500_tickers_filtered
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

def prefetch_ticker_prices(ticker, start_date="2021-04-01", end_date=None):
    """Prefetch price data for a single ticker"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
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
        
        # Save to cache with explicit cleanup
        with DiskCache() as cache:
            cache.set_prices(ticker, cache_data)
        
        date_range = f"{hist.index.min().strftime('%Y-%m-%d')} to {hist.index.max().strftime('%Y-%m-%d')}"
        logger.info(f"Cached {len(cache_data)} price records for {ticker} ({date_range})")
        return True
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return False

def main():
    """Main function to prefetch all filtered S&P 500 ticker prices and bond ETFs"""
    logger.info("Starting price data prefetch for filtered S&P 500 tickers and bond ETFs")
    
    # Get filtered tickers (removes problematic/delisted tickers)
    tickers = get_sp500_tickers_filtered(years_back=4)
    
    # Add bond ETFs used by the backtester
    bond_etfs = ["SHY", "TLT"]
    logger.info(f"Adding bond ETFs to prefetch: {bond_etfs}")
    
    # Combine S&P 500 tickers with bond ETFs
    all_tickers = tickers + bond_etfs
    
    if not all_tickers:
        logger.error("No tickers found to prefetch")
        return
    
    logger.info(f"Found {len(tickers)} filtered S&P 500 tickers + {len(bond_etfs)} bond ETFs = {len(all_tickers)} total tickers to prefetch")
    
    # Track progress
    successful = 0
    failed = 0
    
    # Process each ticker
    for i, ticker in enumerate(all_tickers, 1):
        ticker_type = "Bond ETF" if ticker in bond_etfs else "Stock"
        logger.info(f"[{i:3d}/{len(all_tickers)}] Processing {ticker} ({ticker_type})")
        
        if prefetch_ticker_prices(ticker):
            successful += 1
        else:
            failed += 1
    
    # Summary
    logger.info(f"Prefetch completed: {successful} successful, {failed} failed")
    logger.info(f"Stocks processed: {len(tickers)}, Bond ETFs processed: {len(bond_etfs)}")
    logger.info(f"Price data cached for date range 2021-04-01 to {datetime.now().strftime('%Y-%m-%d')} (or earliest available)")

if __name__ == "__main__":
    main()