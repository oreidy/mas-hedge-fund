#!/usr/bin/env python3
"""
Temporary debug script to check ticker filtering issues.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime
import yfinance as yf
import pandas as pd
from utils.tickers import get_eligible_tickers_for_date, get_sp500_tickers_with_dates_filtered, get_exchange_listing_dates

def check_ticker_filtering():
    """Check why problematic tickers are getting through the filter"""
    
    target_date = "2022-01-01"
    problematic_tickers = ['INFO', 'AMTM', 'CEG', 'FTRE', 'GEHC', 'GEV', 'KVUE', 'LB', 'BAC']
    
    print(f"=== CHECKING TICKER FILTERING FOR {target_date} ===\n")
    
    # 1. Check if INFO is filtered out
    print("1. Checking S&P 500 membership and filtering...")
    sp500_data = get_sp500_tickers_with_dates_filtered(years_back=4)
    
    for ticker in problematic_tickers:
        if ticker in sp500_data:
            dates = sp500_data[ticker]
            print(f"   {ticker}: S&P 500 from {dates['sp500_start']} to {dates['sp500_end']}")
        else:
            print(f"   {ticker}: NOT in S&P 500 data (properly filtered)")
    
    print()
    
    # 2. Check IPO dates
    print("2. Checking IPO dates...")
    present_tickers = [t for t in problematic_tickers if t in sp500_data]
    
    if present_tickers:
        ipo_dates = get_exchange_listing_dates(present_tickers)
        
        for ticker in present_tickers:
            if ticker in ipo_dates:
                ipo_date = ipo_dates[ticker]
                target_dt = datetime.strptime(target_date, "%Y-%m-%d")
                days_listed = (target_dt - ipo_date).days
                print(f"   {ticker}: IPO on {ipo_date.strftime('%Y-%m-%d')} ({days_listed} days before {target_date})")
            else:
                print(f"   {ticker}: NO IPO date found (will use S&P 500 start date)")
    
    print()
    
    # 3. Check eligible tickers
    print("3. Checking final eligible tickers...")
    eligible = get_eligible_tickers_for_date(target_date, min_days_listed=200)
    
    for ticker in problematic_tickers:
        if ticker in eligible:
            print(f"   {ticker}: ❌ WRONGLY INCLUDED in eligible tickers")
        else:
            print(f"   {ticker}: ✅ Correctly excluded from eligible tickers")
    
    print(f"\nTotal eligible tickers: {len(eligible)}")
    print()

def check_yfinance_volume():
    """Check what yfinance returns for volume data for problematic tickers"""
    
    print("=== CHECKING YFINANCE VOLUME DATA ===\n")
    
    problematic_tickers = ['INFO', 'AMTM', 'CEG', 'FTRE', 'GEHC', 'GEV', 'KVUE', 'LB', 'BAC']
    start_date = "2021-12-30"
    end_date = "2022-01-03"
    
    for ticker in problematic_tickers:
        print(f"Checking {ticker} from {start_date} to {end_date}:")
        
        try:
            # Download data
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=20)
            
            if data.empty:
                print(f"   ❌ No data returned")
            else:
                print(f"   📊 Data shape: {data.shape}")
                print(f"   📅 Date range: {data.index.min()} to {data.index.max()}")
                print(data)
                
                # Check volume specifically
                if 'Volume' in data.columns:
                    volume_data = data['Volume']
                    nan_count = volume_data.isna().sum()
                    total_count = len(volume_data)
                    
                    print(f"   📈 Volume: {nan_count}/{total_count} NaN values")
                    
                    if nan_count > 0:
                        print(f"   🔍 Volume data preview:")
                        for date, vol in volume_data.items():
                            vol_str = "NaN" if pd.isna(vol) else f"{vol:,.0f}"
                            print(f"      {date.strftime('%Y-%m-%d')}: {vol_str}")
                    else:
                        print(f"   ✅ All volume data is valid")
                        print(f"   🔍 Sample volumes: {volume_data.tolist()}")
                else:
                    print(f"   ❌ No Volume column found")
                    print(f"   📋 Available columns: {list(data.columns)}")
                    
        except Exception as e:
            print(f"   💥 Error downloading data: {e}")
        
        print()

def main():
    """Run all diagnostic checks"""
    check_ticker_filtering()
    check_yfinance_volume()
    
    print("=== SUMMARY ===")
    print("If INFO appears in eligible tickers, there's a filtering bug.")
    print("If other tickers appear, check their IPO dates vs S&P 500 start dates.")
    print("Volume NaN issues indicate data quality problems from yfinance.")

if __name__ == "__main__":
    main()