#!/usr/bin/env python3
"""
Test that we're not disrupting active cache usage
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.disk_cache import DiskCache

def test_cache_safety():
    print("=== Testing cache safety ===")
    
    # Test 1: Can we use cache after close()?
    print("\n1. Testing cache usage after close()...")
    cache = DiskCache()
    
    # Store some data
    cache.set_prices("AAPL", [{"time": "2025-01-01", "close": 150.0}])
    print("✓ Stored data successfully")
    
    # Retrieve data
    data = cache.get_prices("AAPL")
    print(f"✓ Retrieved data: {len(data)} items")
    
    # Close the cache
    cache.close()
    print("✓ Called close()")
    
    # Try to use after close
    try:
        data_after_close = cache.get_prices("AAPL")
        print(f"✓ Still works after close(): {len(data_after_close)} items")
    except Exception as e:
        print(f"✗ Error after close(): {e}")
    
    # Try to store after close
    try:
        cache.set_prices("MSFT", [{"time": "2025-01-01", "close": 400.0}])
        print("✓ Can still store data after close()")
    except Exception as e:
        print(f"✗ Cannot store after close(): {e}")
    
    print("\n2. Testing global cache behavior...")
    # Test the global cache instance used by API
    from tools.api import _cache as global_cache
    
    # Store something in global cache
    global_cache.set_prices("GLOBAL_TEST", [{"time": "2025-01-01", "close": 200.0}])
    data = global_cache.get_prices("GLOBAL_TEST")
    print(f"✓ Global cache works: {len(data)} items")
    
    # Don't close global cache here - it should only be closed at program exit
    print("✓ Global cache left open for continued use")
    
    print("\n3. Testing multiple cache instances...")
    # This tests the scenario from your backtester
    cache1 = DiskCache()
    cache2 = DiskCache()
    
    cache1.set_prices("CACHE1", [{"time": "2025-01-01", "close": 100.0}])
    cache2.set_prices("CACHE2", [{"time": "2025-01-01", "close": 200.0}])
    
    # Close cache1 but keep cache2 open
    cache1.close()
    
    # cache2 should still work
    data2 = cache2.get_prices("CACHE2")
    print(f"✓ Cache2 still works after Cache1 closed: {len(data2)} items")
    
    # Close cache2
    cache2.close()
    print("✓ All test caches closed safely")

if __name__ == "__main__":
    test_cache_safety()