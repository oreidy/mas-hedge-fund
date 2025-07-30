#!/usr/bin/env python3
"""
Investigate why file descriptors are still growing
"""
import subprocess
import sys
import os
import time
import gc
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def get_detailed_fd_info():
    """Get detailed info about open file descriptors"""
    try:
        result = subprocess.run(['lsof', '-p', str(os.getpid())], 
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        # Count by type
        fd_types = {}
        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    fd_type = parts[4]  # TYPE column
                    fd_types[fd_type] = fd_types.get(fd_type, 0) + 1
        
        return len(lines) - 1, fd_types  # -1 for header
    except:
        return 0, {}

print("=== Detailed File Descriptor Investigation ===")

# Initial state
initial_count, initial_types = get_detailed_fd_info()
print(f"Initial FD count: {initial_count}")
print(f"Initial FD types: {initial_types}")

# Import DiskCache
from data.disk_cache import DiskCache

print(f"\n--- After importing DiskCache ---")
after_import_count, after_import_types = get_detailed_fd_info()
print(f"FD count: {after_import_count} (growth: {after_import_count - initial_count})")
print(f"FD types: {after_import_types}")

# Create ONE cache instance
print(f"\n--- Creating single DiskCache instance ---")
cache = DiskCache()
after_create_count, after_create_types = get_detailed_fd_info()
print(f"FD count: {after_create_count} (growth: {after_create_count - after_import_count})")
print(f"FD types: {after_create_types}")

# Use the cache
print(f"\n--- Using cache (set/get operations) ---")
cache.set_prices("TEST", [{"time": "2025-01-01", "close": 100}])
_ = cache.get_prices("TEST")
after_use_count, after_use_types = get_detailed_fd_info()
print(f"FD count: {after_use_count} (growth: {after_use_count - after_create_count})")
print(f"FD types: {after_use_types}")

# Explicitly close
print(f"\n--- After explicit close() ---")
cache.close()
after_close_count, after_close_types = get_detailed_fd_info()
print(f"FD count: {after_close_count} (growth: {after_close_count - after_use_count})")
print(f"FD types: {after_close_types}")

# Delete reference and force garbage collection
print(f"\n--- After deleting cache reference and GC ---")
del cache
gc.collect()  # Force garbage collection
time.sleep(0.1)  # Give it a moment
after_gc_count, after_gc_types = get_detailed_fd_info()
print(f"FD count: {after_gc_count} (growth: {after_gc_count - after_close_count})")
print(f"FD types: {after_gc_types}")

print(f"\n=== SUMMARY ===")
print(f"Total growth: {after_gc_count - initial_count}")
print(f"Growth from import: {after_import_count - initial_count}")
print(f"Growth from creation: {after_create_count - after_import_count}")
print(f"Growth from usage: {after_use_count - after_create_count}")
print(f"Change from close(): {after_close_count - after_use_count}")
print(f"Change from GC: {after_gc_count - after_close_count}")