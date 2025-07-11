#!/usr/bin/env python3
"""
Test script to check Alpha Vantage API key status
"""

import os
import requests
from dotenv import load_dotenv

def test_api_key(api_key, key_name):
    """Test if an API key works with a simple request."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "AAPL",
        "limit": 1,
        "apikey": api_key
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if "Note" in data and "API call frequency" in data["Note"]:
            return f"{key_name}: ❌ RATE LIMITED - {data['Note']}"
        elif "Error Message" in data:
            return f"{key_name}: ❌ ERROR - {data['Error Message']}"
        elif "feed" in data or "items" in data:
            return f"{key_name}: ✅ WORKING - {data.get('items', 0)} items available"
        elif "Information" in data:
            return f"{key_name}: ⚠️ RATE LIMITED - {data['Information']}"
        else:
            return f"{key_name}: ⚠️ UNKNOWN RESPONSE - {str(data)[:100]}..."
    
    except Exception as e:
        return f"{key_name}: ❌ REQUEST FAILED - {str(e)}"

def main():
    load_dotenv()
    
    print("Testing Alpha Vantage API Keys")
    print("=" * 50)
    
    # Test all API keys
    keys_to_test = []
    
    # First key
    key1 = os.getenv("ALPHA_VANTAGE_API_KEY")
    if key1:
        keys_to_test.append((key1, "API_KEY_1"))
    
    # Numbered keys
    i = 2
    while i <= 20:
        key = os.getenv(f"ALPHA_VANTAGE_API_KEY_{i}")
        if key and key.strip():
            keys_to_test.append((key, f"API_KEY_{i}"))
        i += 1
    
    print(f"Found {len(keys_to_test)} API keys to test\n")
    
    for api_key, key_name in keys_to_test:
        result = test_api_key(api_key, key_name)
        print(result)
    
    print(f"\nRecommendation:")
    print("If most keys show 'RATE LIMITED', wait 24 hours and try again.")
    print("If keys show 'WORKING', there might be an issue with the prefetch script.")

if __name__ == "__main__":
    main()