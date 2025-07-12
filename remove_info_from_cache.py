#!/usr/bin/env python3
"""
Script to remove INFO ticker from cache databases without deleting everything else.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.disk_cache import DiskCache
from pathlib import Path

def remove_info_from_cache():
    """Remove INFO ticker from all cache databases"""
    
    print("=== REMOVING INFO FROM CACHE DATABASES ===\n")
    
    # Initialize cache
    cache = DiskCache(cache_dir=Path(__file__).parent / "cache")
    
    ticker = "INFO"
    removed_count = 0
    
    # 1. Remove from prices cache
    print("1. Checking prices cache...")
    try:
        # Check if INFO exists in prices cache
        cached_prices = cache.get_prices(ticker)
        if cached_prices:
            print(f"   Found {len(cached_prices)} price entries for {ticker}")
            # Remove from cache
            cache.caches["prices"].delete(f"prices:{ticker}")
            print(f"   ✅ Removed {ticker} from prices cache")
            removed_count += 1
        else:
            print(f"   No price data found for {ticker}")
    except Exception as e:
        print(f"   ❌ Error checking prices cache: {e}")
    
    # 2. Remove from financial metrics cache
    print("\n2. Checking financial metrics cache...")
    try:
        cached_metrics = cache.get_financial_metrics(ticker)
        if cached_metrics:
            print(f"   Found {len(cached_metrics)} financial metric entries for {ticker}")
            cache.caches["financial_metrics"].delete(f"financial_metrics:{ticker}")
            print(f"   ✅ Removed {ticker} from financial metrics cache")
            removed_count += 1
        else:
            print(f"   No financial metrics found for {ticker}")
    except Exception as e:
        print(f"   ❌ Error checking financial metrics cache: {e}")
    
    # 3. Remove from company news cache
    print("\n3. Checking company news cache...")
    try:
        cached_news = cache.get_company_news(ticker)
        if cached_news:
            print(f"   Found {len(cached_news)} news entries for {ticker}")
            cache.caches["company_news"].delete(f"company_news:{ticker}")
            print(f"   ✅ Removed {ticker} from company news cache")
            removed_count += 1
        else:
            print(f"   No company news found for {ticker}")
    except Exception as e:
        print(f"   ❌ Error checking company news cache: {e}")
    
    # 4. Remove from IPO dates cache
    print("\n4. Checking IPO dates cache...")
    try:
        if "ipo_dates" in cache.caches:
            cached_ipo = cache.caches["ipo_dates"].get(f"ipo:{ticker}")
            if cached_ipo:
                print(f"   Found IPO date for {ticker}: {cached_ipo}")
                cache.caches["ipo_dates"].delete(f"ipo:{ticker}")
                print(f"   ✅ Removed {ticker} from IPO dates cache")
                removed_count += 1
            else:
                print(f"   No IPO date found for {ticker}")
        else:
            print("   No IPO dates cache found")
    except Exception as e:
        print(f"   ❌ Error checking IPO dates cache: {e}")
    
    # 5. Remove from outstanding shares cache
    print("\n5. Checking outstanding shares cache...")
    try:
        if "outstanding_shares" in cache.caches:
            # Outstanding shares might have multiple entries for different dates
            # We need to check the cache structure
            shares_cache = cache.caches["outstanding_shares"]
            keys_to_remove = []
            
            # Get all keys that start with the ticker
            for key in shares_cache:
                if key.startswith(f"outstanding_shares:{ticker}:"):
                    keys_to_remove.append(key)
            
            if keys_to_remove:
                print(f"   Found {len(keys_to_remove)} outstanding shares entries for {ticker}")
                for key in keys_to_remove:
                    shares_cache.delete(key)
                print(f"   ✅ Removed {len(keys_to_remove)} outstanding shares entries for {ticker}")
                removed_count += len(keys_to_remove)
            else:
                print(f"   No outstanding shares found for {ticker}")
        else:
            print("   No outstanding shares cache found")
    except Exception as e:
        print(f"   ❌ Error checking outstanding shares cache: {e}")
    
    # 6. Remove from line items cache
    print("\n6. Checking line items cache...")
    try:
        if "line_items" in cache.caches:
            line_items_cache = cache.caches["line_items"]
            keys_to_remove = []
            
            # Get all keys that start with the ticker
            for key in line_items_cache:
                if key.startswith(f"line_items:{ticker}:"):
                    keys_to_remove.append(key)
            
            if keys_to_remove:
                print(f"   Found {len(keys_to_remove)} line items entries for {ticker}")
                for key in keys_to_remove:
                    line_items_cache.delete(key)
                print(f"   ✅ Removed {len(keys_to_remove)} line items entries for {ticker}")
                removed_count += len(keys_to_remove)
            else:
                print(f"   No line items found for {ticker}")
        else:
            print("   No line items cache found")
    except Exception as e:
        print(f"   ❌ Error checking line items cache: {e}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total cache entries removed for {ticker}: {removed_count}")
    
    if removed_count > 0:
        print(f"✅ Successfully cleaned {ticker} from cache databases")
        print("You can now run your backtester - INFO should be properly excluded")
    else:
        print(f"ℹ️  No cache entries found for {ticker} (already clean)")

def verify_removal():
    """Verify that INFO has been removed from cache"""
    print("\n=== VERIFICATION ===")
    
    cache = DiskCache(cache_dir=Path(__file__).parent / "cache")
    ticker = "INFO"
    
    # Check each cache type
    checks = [
        ("prices", cache.get_prices(ticker)),
        ("financial_metrics", cache.get_financial_metrics(ticker)),
        ("company_news", cache.get_company_news(ticker)),
    ]
    
    all_clean = True
    for cache_type, data in checks:
        if data:
            print(f"❌ {cache_type}: Still has data for {ticker}")
            all_clean = False
        else:
            print(f"✅ {cache_type}: Clean (no data for {ticker})")
    
    if all_clean:
        print(f"\n🎉 Verification successful: {ticker} completely removed from cache")
    else:
        print(f"\n⚠️  Some cache entries may still exist for {ticker}")

if __name__ == "__main__":
    remove_info_from_cache()
    verify_removal()