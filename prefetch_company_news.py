#!/usr/bin/env python3
"""
Script to prefetch company news from Alpha Vantage API.
Downloads 5 years of news data for S&P 500 companies using multiple API keys.
"""

import os
import json
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.tickers import get_sp500_tickers

def load_alpha_vantage_keys():
    """Load all Alpha Vantage API keys from environment variables."""
    keys = []
    i = 1
    while True:
        key_name = f"ALPHA_VANTAGE_API_KEY_{i}" if i > 1 else "ALPHA_VANTAGE_API_KEY"
        key = os.getenv(key_name)
        if key and key.strip():
            keys.append(key.strip())
            i += 1
        else:
            break
    return keys

def get_news_data(ticker, api_key, time_from, time_to):
    """
    Get the 1000 most relevant news articles for a ticker in the last 5 years.
    Makes exactly one API request per ticker.
    
    Args:
        ticker: Stock ticker symbol
        api_key: Alpha Vantage API key
        time_from: Start time in YYYYMMDDTHHMM format
        time_to: End time in YYYYMMDDTHHMM format
    
    Returns:
        dict: API response data or None if failed
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from,
        "time_to": time_to,
        "sort": "RELEVANCE",
        "limit": 1000,
        "apikey": api_key
    }
    
    print(f"Making single request for {ticker}: getting 1000 most relevant articles from last 5 years")
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            print(f"API Error for {ticker}: {data['Error Message']}")
            return None
        if "Note" in data and "API call frequency" in data["Note"]:
            print(f"Rate limit reached for {ticker}")
            return "RATE_LIMIT"
        if "Information" in data and "rate limit" in data["Information"]:
            print(f"Rate limit reached for {ticker}")
            return "RATE_LIMIT"
        
        return data
    except requests.RequestException as e:
        print(f"Request failed for {ticker}: {e}")
        return None

def save_news_data(ticker, data, cache_dir):
    """Save news data to cache directory."""
    ticker_file = cache_dir / f"{ticker}_news.json"
    
    with open(ticker_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved news data for {ticker} to {ticker_file}")

def get_date_ranges_for_5_years():
    """Get date ranges to cover 5 years of data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    time_from = start_date.strftime("%Y%m%dT0000")
    time_to = end_date.strftime("%Y%m%dT2359")
    
    return time_from, time_to

def main():
    print("Starting company news prefetch...")
    print("Using respectful delays: 5 seconds between requests, 60 seconds between API keys")
    
    # Load environment variables
    load_dotenv()
    
    # Get S&P 500 tickers
    print("Loading S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} S&P 500 tickers")
    
    # Load API keys - full deployment
    api_keys = load_alpha_vantage_keys()
    print(f"Found {len(api_keys)} Alpha Vantage API keys - FULL DEPLOYMENT MODE")
    
    if not api_keys:
        print("No Alpha Vantage API keys found!")
        return
    
    # Setup cache directory
    cache_dir = Path("cache/company_news")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Get date range for 5 years
    time_from, time_to = get_date_ranges_for_5_years()
    print(f"Downloading news from {time_from} to {time_to}")
    
    print("Starting in 3 seconds...")
    time.sleep(3)
    
    total_tickers_processed = 0
    
    # Process each API key
    for key_index, api_key in enumerate(api_keys, 1):
        print(f"\n--- Using API Key {key_index}/{len(api_keys)} ---")
        
        requests_used = 0
        max_requests = 25  # Alpha Vantage daily limit
        tickers_processed_this_key = 0
        
        # Process tickers starting from where we left off
        remaining_tickers = tickers[total_tickers_processed:]
        
        for ticker in remaining_tickers:
            if requests_used >= max_requests:
                print(f"Reached daily limit for API key {key_index}")
                break
            
            print(f"\nProcessing {ticker} (Request {requests_used + 1}/{max_requests})")
            
            # Check if we already have data for this ticker
            ticker_file = cache_dir / f"{ticker}_news.json"
            if ticker_file.exists():
                print(f"Skipping {ticker} - already exists in cache")
                tickers_processed_this_key += 1
                total_tickers_processed += 1
                continue
            
            # Make single request for this ticker - get 1000 most relevant articles from last 5 years
            data = get_news_data(ticker, api_key, time_from, time_to)
            
            if data == "RATE_LIMIT":
                print(f"Rate limit reached for API key {key_index}")
                break
            elif data is None:
                print(f"Failed to get data for {ticker}")
                # Still count as a request
                requests_used += 1
                continue
            
            # Save the data
            save_news_data(ticker, data, cache_dir)
            
            # Count articles and date range
            if "feed" in data and data["feed"]:
                articles_count = len(data["feed"])
                dates = [article.get("time_published", "") for article in data["feed"]]
                dates = [d for d in dates if d]  # Remove empty dates
                if dates:
                    min_date = min(dates)
                    max_date = max(dates)
                    print(f"  ✓ Downloaded {articles_count} articles from {min_date} to {max_date}")
                else:
                    print(f"  ✓ Downloaded {articles_count} articles (no date info)")
            else:
                print(f"  ✓ No articles found for {ticker}")
            
            requests_used += 1
            tickers_processed_this_key += 1
            total_tickers_processed += 1
            
            print(f"  Moving to next ticker...")
            
            # Respectful delay between requests (5 seconds)
            print("    Waiting 5 seconds before next request...")
            time.sleep(5)
        
        print(f"API Key {key_index} processed {tickers_processed_this_key} tickers")
        
        # If we've processed all tickers, break
        if total_tickers_processed >= len(tickers):
            print("All tickers have been processed!")
            break
        
        # Wait before switching to next API key to avoid abuse detection
        if key_index < len(api_keys):  # Don't wait after the last key
            print(f"Waiting 60 seconds before switching to next API key...")
            time.sleep(60)
    
    print(f"\n--- Final Summary ---")
    print(f"Total tickers processed: {total_tickers_processed}/{len(tickers)}")
    print(f"Data saved to: {cache_dir}")
    
    # Show some statistics
    json_files = list(cache_dir.glob("*_news.json"))
    print(f"Total news files in cache: {len(json_files)}")
    
    if json_files:
        print("\nSample of cached tickers:")
        for f in json_files[:10]:
            ticker = f.stem.replace("_news", "")
            print(f"  {ticker}")

if __name__ == "__main__":
    main()