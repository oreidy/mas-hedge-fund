#!/usr/bin/env python3
"""
Script to clear only the prices cache to force re-processing with new MultiIndex handling.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.disk_cache import DiskCache
from pathlib import Path

def clear_prices_cache():
    """Clear only the prices cache"""
    
    print("=== CLEARING PRICES CACHE ===\n")
    
    # Initialize cache
    cache = DiskCache(cache_dir=Path(__file__).parent / "cache")
    
    try:
        # Get the prices cache
        prices_cache = cache.caches["prices"]
        
        # Count existing entries
        original_count = len(prices_cache)
        print(f"Found {original_count} entries in prices cache")
        
        if original_count > 0:
            # Clear all prices cache entries
            prices_cache.clear()
            
            # Verify it's empty
            new_count = len(prices_cache)
            print(f"✅ Cleared prices cache: {original_count} → {new_count} entries")
            
            if new_count == 0:
                print("🎉 Prices cache successfully cleared!")
                print("\nNow run your backtester - it will re-download and process data with proper MultiIndex handling.")
            else:
                print("⚠️  Cache may not be completely clear")
        else:
            print("ℹ️  Prices cache was already empty")
            
    except Exception as e:
        print(f"❌ Error clearing prices cache: {e}")

if __name__ == "__main__":
    clear_prices_cache()