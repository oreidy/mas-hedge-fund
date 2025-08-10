#!/usr/bin/env python3
"""
Script to prefetch company news for all filtered tickers.
This creates a comprehensive database of company news from the last 4 years
to eliminate time-consuming downloads during analysis.

Two modes:
1. populate_historical_company_news() - Initial historical data population (4 years)
2. prefetch_all_company_news() - Regular updates for recent news

Uses Alpha Vantage premium API with 75 requests per minute limit.
Since API only returns ~600-700 most recent news items per request,
historical population loops through dates to get complete coverage.

NOTE: This script has a separate populate function unlike prefetch_insider_trades.py
because Alpha Vantage API has pagination limitations - it only returns the most recent
~600-700 news items regardless of date range. To get complete historical coverage,
we need to make multiple requests per ticker, using the earliest date from each
response as the end_date for the next request. In contrast, SEC EDGAR insider trades
API returns all trades within a specified date range in a single request, so no
separate populate function is needed.

Usage:
    python prefetch_company_news.py              # Regular updates
    python prefetch_company_news.py populate     # Initial historical population
    python prefetch_company_news.py populate AAPL # Resume from specific ticker
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Optional
import time

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tickers import get_sp500_tickers_filtered
from tools.api import get_company_news
from data.models import CompanyNews
from data.company_news_db import (
    setup_database,
    save_company_news,
    get_news_database_stats,
    get_news_tickers_needing_update,
    get_global_most_recent_news_date,
    get_company_news as get_company_news_from_db
)
from utils.logger import logger


def get_earliest_news_date(ticker: str) -> Optional[str]:
    """
    Get the earliest news date for a ticker from the database.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Earliest news date as YYYY-MM-DD string, or None if no news found
    """
    news_items = get_company_news_from_db(ticker)
    if not news_items:
        return None
    
    # News items are ordered by date descending, so get the last one
    earliest_date = min(item.date for item in news_items)
    return earliest_date


def populate_historical_company_news(years_back: int = 4, max_retries: int = 50, start_ticker: str = None):
    """
    Initial population function to download historical company news.
    This function loops through tickers and dates to get complete historical data.
    
    Args:
        years_back: Number of years of historical data to fetch
        max_retries: Maximum number of date iterations per ticker 
        start_ticker: Ticker symbol to start from (useful for resuming after interruption)
    """
    logger.info(f"Starting historical population of company news for {years_back} years")
    
    # Setup database
    setup_database()
    
    # Check for API key
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.error("No Alpha Vantage API key found in environment variable ALPHA_VANTAGE_API_KEY")
        return
    
    logger.info("Found Alpha Vantage API key")
    
    # Get all filtered tickers
    all_tickers = get_sp500_tickers_filtered()
    logger.info(f"Found {len(all_tickers)} filtered tickers")
    
    # Calculate date range for historical data
    start_date = (datetime.now() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
    logger.info(f"Target historical date range: {start_date} to present")
    
    # Process each ticker individually
    total_tickers = len(all_tickers)
    
    # Handle start_ticker parameter for resuming
    start_index = 0
    if start_ticker:
        if start_ticker in all_tickers:
            start_index = all_tickers.index(start_ticker)
            logger.info(f"Resuming from ticker {start_ticker} (index {start_index})")
        else:
            logger.warning(f"Start ticker {start_ticker} not found in ticker list. Starting from beginning.")
    
    for ticker_idx, ticker in enumerate(all_tickers):
        # Skip tickers before start_ticker
        if ticker_idx < start_index:
            logger.info(f"Skipping ticker {ticker} ({ticker_idx + 1}/{total_tickers}) - before start ticker")
            continue
            
        logger.info(f"Processing ticker {ticker} ({ticker_idx + 1}/{total_tickers})")
        
        # Check if we already have sufficient historical data
        earliest_date = get_earliest_news_date(ticker)
        if earliest_date and earliest_date <= start_date:
            logger.info(f"✓ {ticker}: Already has historical data back to {earliest_date}")
            continue
        
        # Fetch historical data in chunks
        current_end_date = datetime.now().strftime("%Y-%m-%d")
        target_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        iteration_count = 0
        while iteration_count < max_retries:
            try:
                # Fetch news for current date range
                logger.info(f"Fetching {ticker} news ending {current_end_date}")
                
                news_dict = get_company_news(
                    [ticker],
                    end_date=current_end_date,
                    start_date=None,  # Let API determine start date
                    limit=1000,
                    verbose_data=False
                )
                
                news_list = news_dict.get(ticker, [])
                
                if not news_list:
                    logger.info(f"No news found for {ticker} ending {current_end_date}")
                    break
                
                # Save to database
                saved_count = save_company_news(ticker, news_list)
                logger.info(f"✓ {ticker}: {saved_count} new news items saved (total fetched: {len(news_list)})")
                
                # Check if we have enough historical data
                earliest_fetched = min(item.date for item in news_list)
                earliest_fetched_dt = datetime.strptime(earliest_fetched, "%Y-%m-%d")
                
                if earliest_fetched_dt <= target_start_date:
                    logger.info(f"✓ {ticker}: Reached target historical date {earliest_fetched}")
                    break
                
                # Set next end date to day before earliest fetched news
                next_end_date = (earliest_fetched_dt - timedelta(days=1)).strftime("%Y-%m-%d")
                
                # Check if we're making progress
                if next_end_date == current_end_date:
                    logger.warning(f"No progress for {ticker}, stopping")
                    break
                
                current_end_date = next_end_date
                iteration_count += 1
                
                # Rate limiting - wait between requests
                time.sleep(2)  # Conservative rate limiting
                
            except Exception as e:
                iteration_count += 1
                logger.error(f"Error fetching historical news for {ticker} (iteration {iteration_count}): {e}")
                if iteration_count < max_retries:
                    time.sleep(10)  # Wait before retry
                else:
                    logger.error(f"Max retries reached for {ticker}")
                    break
    
    # Final statistics
    final_stats = get_news_database_stats()
    logger.info(f"✅ Historical news population completed!")
    logger.info(f"📊 Final Statistics:")
    logger.info(f"   • Total news in database: {final_stats['total_news']}")
    logger.info(f"   • Total tickers: {final_stats['unique_tickers']}")
    logger.info(f"   • Database size: {final_stats['database_size_mb']} MB")


def prefetch_all_company_news(max_retries: int = 3):
    """
    Update company news for all filtered tickers.
    Only downloads recent news since the last update.
    Use populate_historical_company_news() for initial 4-year data population.
    
    Args:
        max_retries: Maximum number of retries for failed tickers
    """
    logger.info("Starting prefetch of company news for filtered tickers")
    
    # Setup database
    setup_database()
    
    # Check for API key
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.error("No Alpha Vantage API key found in environment variable ALPHA_VANTAGE_API_KEY")
        return
    
    logger.info("Found Alpha Vantage API key")
    
    # Get current database stats
    stats = get_news_database_stats()
    logger.info(f"Database currently has {stats['total_news']} news items for {stats['unique_tickers']} tickers")
    if stats['sentiment_breakdown']:
        logger.info(f"Sentiment breakdown: {stats['sentiment_breakdown']}")
    
    # Get all filtered tickers
    all_tickers = get_sp500_tickers_filtered()
    logger.info(f"Found {len(all_tickers)} filtered tickers")
    
    # Get tickers needing update (max 1 day old)
    tickers_to_update = get_news_tickers_needing_update(all_tickers, max_age_days=1)
    
    # Use global most recent news date for efficiency
    global_most_recent_news = get_global_most_recent_news_date()
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    if global_most_recent_news:
        # Start from day after the most recent news across all tickers
        start_date_dt = datetime.strptime(global_most_recent_news, "%Y-%m-%d") + timedelta(days=1)
        start_date = start_date_dt.strftime("%Y-%m-%d")
        logger.info(f"Using global date range: {start_date} to {end_date} (last news across all tickers: {global_most_recent_news})")
    else:
        # No previous news at all, recommend using populate_historical_company_news
        logger.warning("No previous news found. Use populate_historical_company_news() for initial data population.")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")  # Just get last week
        logger.info(f"Fetching recent news from {start_date} to {end_date}")
    
    # Log status
    up_to_date_count = len(all_tickers) - len(tickers_to_update)
    logger.info(f"Tickers up to date: {up_to_date_count}")
    logger.info(f"Tickers needing update: {len(tickers_to_update)}")
    
    if not tickers_to_update:
        logger.info("All tickers are up to date!")
        return
    
    logger.info(f"Found {len(tickers_to_update)} tickers needing updates")
    
    # Process tickers individually (no batching since API calls are per ticker anyway)
    failed_tickers = []
    total_tickers = len(tickers_to_update)
    
    for i, ticker in enumerate(tickers_to_update):
        logger.info(f"Processing ticker {ticker} ({i + 1}/{total_tickers})")
        
        try:
            news_dict = get_company_news(
                [ticker],
                end_date=end_date,
                start_date=start_date,
                limit=1000,
                verbose_data=False
            )
            
            # Save to database
            news_list = news_dict.get(ticker, [])
            saved_count = save_company_news(ticker, news_list)
            logger.info(f"✓ {ticker}: {saved_count} new news items saved (total fetched: {len(news_list)})")
            
            # Rate limiting between requests
            if i < total_tickers - 1:  # Don't delay after last ticker
                time.sleep(1)  # Conservative rate limiting
            
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {e}")
            failed_tickers.append(ticker)
    
    # Retry failed tickers individually
    if failed_tickers:
        logger.info(f"Retrying {len(failed_tickers)} failed tickers...")
        
        for ticker in failed_tickers:
            for retry in range(max_retries):
                try:
                    logger.info(f"Retrying {ticker} (attempt {retry + 1}/{max_retries})")
                    
                    news_dict = get_company_news(
                        [ticker],
                        end_date=end_date,
                        start_date=start_date,
                        limit=1000,
                        verbose_data=False
                    )
                    
                    news_list = news_dict.get(ticker, [])
                    saved_count = save_company_news(ticker, news_list)
                    logger.info(f"✓ {ticker}: {saved_count} new news items saved (total fetched: {len(news_list)})")
                    break
                    
                except Exception as e:
                    logger.warning(f"Retry {retry + 1} failed for {ticker}: {e}")
                    if retry < max_retries - 1:
                        time.sleep(10)  # Wait before retry
    
    # Final statistics
    final_stats = get_news_database_stats()
    new_news = final_stats['total_news'] - stats['total_news']
    
    logger.info(f"✅ Company news prefetch completed!")
    logger.info(f"📊 Statistics:")
    logger.info(f"   • Total news in database: {final_stats['total_news']} (+{new_news})")
    logger.info(f"   • Total tickers processed: {final_stats['unique_tickers']}")
    logger.info(f"   • Recent news (30d): {final_stats['recent_news_30d']}")
    logger.info(f"   • Sentiment breakdown: {final_stats['sentiment_breakdown']}")
    logger.info(f"   • Database size: {final_stats['database_size_mb']} MB")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "populate":
        # Run initial historical population
        start_ticker = sys.argv[2] if len(sys.argv) > 2 else None
        if start_ticker:
            logger.info(f"Starting historical population from ticker: {start_ticker}")
        populate_historical_company_news(years_back=4, start_ticker=start_ticker)
    else:
        # Run regular update
        prefetch_all_company_news()