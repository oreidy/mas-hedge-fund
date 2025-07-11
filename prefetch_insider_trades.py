#!/usr/bin/env python3
"""
Script to prefetch insider trades for all S&P 500 tickers.
This creates a comprehensive database of insider trades from the last 5 years
to eliminate time-consuming downloads during analysis.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set
import time

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.tickers import get_sp500_tickers
from tools.sec_edgar_scraper import fetch_multiple_insider_trades
from data.models import InsiderTrade
from data.insider_trades_db import (
    setup_database,
    save_insider_trades,
    get_database_stats,
    get_tickers_needing_update,
    get_most_recent_trade_dates
)
from utils.logger import logger




def prefetch_all_insider_trades(batch_size: int = 10, max_retries: int = 3):
    """
    Prefetch insider trades for all S&P 500 tickers from the last 5 years.
    Only downloads missing or outdated data.
    
    Args:
        batch_size: Number of tickers to process in each batch
        max_retries: Maximum number of retries for failed tickers
    """
    logger.info("Starting prefetch of S&P 500 insider trades")
    
    # Setup database
    setup_database()
    
    # Get current database stats
    stats = get_database_stats()
    logger.info(f"Database currently has {stats['total_trades']} trades for {stats['unique_tickers']} tickers")
    
    # Get all S&P 500 tickers
    all_tickers = get_sp500_tickers()
    logger.info(f"Found {len(all_tickers)} S&P 500 tickers")
    
    # Get tickers needing update (max 1 day old)
    tickers_to_update = get_tickers_needing_update(all_tickers, max_age_days=1)
    
    # Get most recent trade dates for tickers that need updates
    recent_trade_dates = get_most_recent_trade_dates(tickers_to_update)
    
    # Calculate dynamic date ranges for each ticker
    end_date = datetime.now().strftime("%Y-%m-%d")
    ticker_date_ranges = {}
    
    for ticker in tickers_to_update:
        most_recent_trade = recent_trade_dates.get(ticker)
        if most_recent_trade:
            # Start from day after most recent trade
            start_date_dt = datetime.strptime(most_recent_trade, "%Y-%m-%d") + timedelta(days=1)
            start_date = start_date_dt.strftime("%Y-%m-%d")
            logger.info(f"{ticker}: Fetching from {start_date} to {end_date} (last trade: {most_recent_trade})")
        else:
            # No previous trades, fetch last 5 years
            start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
            logger.info(f"{ticker}: No previous trades, fetching from {start_date} to {end_date}")
        
        ticker_date_ranges[ticker] = (start_date, end_date)
    
    # Log status
    up_to_date_count = len(all_tickers) - len(tickers_to_update)
    logger.info(f"Tickers up to date: {up_to_date_count}")
    logger.info(f"Tickers needing update: {len(tickers_to_update)}")
    
    if not tickers_to_update:
        logger.info("All tickers are up to date!")
        return
    
    logger.info(f"Found {len(tickers_to_update)} tickers needing updates")
    
    # Process tickers in batches
    failed_tickers = []
    total_batches = (len(tickers_to_update) + batch_size - 1) // batch_size
    processed_count = 0
    
    for i in range(0, len(tickers_to_update), batch_size):
        batch = tickers_to_update[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({processed_count}/{len(tickers_to_update)} tickers completed)")
        logger.info(f"Current batch: {batch}")
        
        try:
            # Fetch insider trades for this batch with individual date ranges
            trades_dict = {}
            for ticker in batch:
                start_date, end_date = ticker_date_ranges[ticker]
                logger.info(f"Fetching {ticker} from {start_date} to {end_date}")
                
                ticker_trades = fetch_multiple_insider_trades(
                    tickers=[ticker],
                    start_date=start_date,
                    end_date=end_date,
                    verbose_data=True
                )
                trades_dict.update(ticker_trades)
            
            # Save to database
            for ticker in batch:
                trades = trades_dict.get(ticker, [])
                saved_count = save_insider_trades(ticker, trades)
                logger.info(f"✓ {ticker}: {saved_count} new trades saved (total: {len(trades)})")
                processed_count += 1
            
            # Brief pause between batches to be respectful to SEC servers
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            failed_tickers.extend(batch)
            processed_count += len(batch)  # Count as processed even if failed
    
    # Retry failed tickers individually
    if failed_tickers:
        logger.info(f"Retrying {len(failed_tickers)} failed tickers individually...")
        
        for ticker in failed_tickers:
            for retry in range(max_retries):
                try:
                    logger.info(f"Retrying {ticker} (attempt {retry + 1}/{max_retries})")
                    start_date, end_date = ticker_date_ranges[ticker]
                    trades_dict = fetch_multiple_insider_trades(
                        tickers=[ticker],
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    trades = trades_dict.get(ticker, [])
                    saved_count = save_insider_trades(ticker, trades)
                    logger.info(f"✓ {ticker}: {saved_count} new trades saved (total: {len(trades)})")
                    break
                    
                except Exception as e:
                    logger.warning(f"Retry {retry + 1} failed for {ticker}: {e}")
                    if retry < max_retries - 1:
                        time.sleep(5)  # Wait before retry
    
    # Final statistics
    final_stats = get_database_stats()
    new_trades = final_stats['total_trades'] - stats['total_trades']
    
    logger.info(f"✅ Prefetch completed!")
    logger.info(f"📊 Statistics:")
    logger.info(f"   • Total trades in database: {final_stats['total_trades']} (+{new_trades})")
    logger.info(f"   • Total tickers processed: {final_stats['unique_tickers']}")
    logger.info(f"   • Database size: {final_stats['database_size_mb']} MB")

if __name__ == "__main__":
    prefetch_all_insider_trades(batch_size=5)  # Start with smaller batches