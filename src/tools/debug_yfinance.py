#!/usr/bin/env python3
"""
Debug script to check yfinance availability for specific tickers
"""
import yfinance as yf
import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_ticker(ticker: str, start_date: str = "2021-04-01", end_date: str = "2025-07-01"):
    """Debug a specific ticker to check data availability"""
    print(f"\n{'='*50}")
    print(f"Debugging ticker: {ticker}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*50}")
    
    try:
        # Create yfinance Ticker object
        stock = yf.Ticker(ticker)
        
        # Check basic info
        print("\n1. Basic ticker info:")
        try:
            info = stock.info
            print(f"   Company Name: {info.get('longName', 'N/A')}")
            print(f"   Sector: {info.get('sector', 'N/A')}")
            print(f"   Exchange: {info.get('exchange', 'N/A')}")
            print(f"   Currency: {info.get('currency', 'N/A')}")
            if 'firstTradeDateEpochUtc' in info and info['firstTradeDateEpochUtc']:
                first_trade = datetime.fromtimestamp(info['firstTradeDateEpochUtc'])
                print(f"   First Trade Date: {first_trade.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"   Error getting info: {e}")
        
        # Check historical data
        print("\n2. Historical data availability:")
        try:
            hist = stock.history(start=start_date, end=end_date)
            print(f"   Total rows: {len(hist)}")
            if not hist.empty:
                print(f"   Date range: {hist.index.min().strftime('%Y-%m-%d')} to {hist.index.max().strftime('%Y-%m-%d')}")
                print(f"   Columns: {list(hist.columns)}")
                
                # Check for missing data
                print("\n   Data quality check:")
                for col in hist.columns:
                    nan_count = hist[col].isna().sum()
                    print(f"     {col}: {nan_count} NaN values out of {len(hist)}")
                
                # Show first few rows
                print(f"\n   First 5 rows:")
                print(hist.head())
                
                # Show last few rows
                print(f"\n   Last 5 rows:")
                print(hist.tail())
                
            else:
                print("   No historical data found!")
                
        except Exception as e:
            print(f"   Error getting historical data: {e}")
        
        # Check specific problematic date
        print("\n3. Checking specific date (2021-04-01):")
        try:
            single_day = stock.history(start="2021-04-01", end="2021-04-02")
            if not single_day.empty:
                print(f"   Data found for 2021-04-01:")
                print(single_day)
            else:
                print("   No data found for 2021-04-01")
        except Exception as e:
            print(f"   Error checking 2021-04-01: {e}")
        
        # Check recent data
        print("\n4. Checking recent data (last 30 days):")
        try:
            recent = stock.history(period="1mo")
            if not recent.empty:
                print(f"   Recent data rows: {len(recent)}")
                print(f"   Recent date range: {recent.index.min().strftime('%Y-%m-%d')} to {recent.index.max().strftime('%Y-%m-%d')}")
            else:
                print("   No recent data found!")
        except Exception as e:
            print(f"   Error getting recent data: {e}")
            
    except Exception as e:
        print(f"General error with ticker {ticker}: {e}")

def main():
    """Main function to debug specific tickers"""
    print("YFinance Ticker Debug Script")
    print("="*60)
    
    # Tickers to debug
    tickers = ["AEP", "AMTM"]
    
    for ticker in tickers:
        debug_ticker(ticker)
    
    print(f"\n{'='*60}")
    print("Debug completed")

if __name__ == "__main__":
    main()