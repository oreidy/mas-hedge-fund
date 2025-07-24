#!/usr/bin/env python3

import sys
sys.path.append('./src')

from data.disk_cache import DiskCache
import pandas as pd

# Initialize cache
cache = DiskCache()

# Test problematic tickers from your error message
problematic_tickers = ["AOS", "APD", "HIG"]

print("=== CHECKING CACHE FOR CORRUPTION ===")

for ticker in problematic_tickers:
    print(f"\nChecking {ticker}:")
    cached_data = cache.get_prices(ticker)
    
    if cached_data:
        print(f"  Found {len(cached_data)} cached records")
        
        # Check for volume issues
        volume_issues = 0
        for record in cached_data:
            if 'volume' not in record or pd.isna(record['volume']) or record['volume'] is None:
                volume_issues += 1
        
        if volume_issues > 0:
            print(f"  ❌ CORRUPTED: {volume_issues}/{len(cached_data)} records have volume issues")
            print(f"  Recommend clearing cache for {ticker}")
        else:
            print(f"  ✅ CLEAN: All records have valid volume data")
    else:
        print(f"  No cached data found for {ticker}")

print(f"\n=== RECOMMENDATION ===")
print("If any tickers show corruption above, clear the price cache:")
print("rm -rf /Users/oliverreidy/mas-hedge-fund/cache/prices/*")