"""
Centralized database utility module for insider trades data.
Handles all SQLite operations for insider trades storage and retrieval.
"""

import sqlite3
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime, date

from .models import InsiderTrade

logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = Path(__file__).parent.parent.parent / "insider_trades_database.db"


def setup_database():
    """Initialize the insider trades database with required tables and indexes."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS insider_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            filing_date DATE,
            transaction_date DATE,
            name TEXT,
            title TEXT,
            issuer TEXT,
            is_board_director BOOLEAN,
            transaction_shares REAL,
            transaction_price_per_share REAL,
            transaction_value REAL,
            shares_owned_before_transaction REAL,
            shares_owned_after_transaction REAL,
            security_title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, filing_date, name, transaction_date, transaction_shares, transaction_price_per_share)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ticker_sync_status (
            ticker TEXT PRIMARY KEY,
            last_sync_date DATE,
            last_trade_date DATE,
            total_trades INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_ticker ON insider_trades(ticker)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_filing_date ON insider_trades(filing_date)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_transaction_date ON insider_trades(transaction_date)
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database setup completed")


def save_insider_trades(ticker: str, trades: List[InsiderTrade]) -> int:
    """
    Save insider trades to the database and update sync status.
    
    Args:
        ticker: Stock ticker symbol
        trades: List of InsiderTrade objects to save
        
    Returns:
        Number of new trades added to database
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    new_trades_count = 0
    latest_trade_date = None
    
    # Process trades if any exist
    for trade in trades:
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO insider_trades (
                    ticker, filing_date, transaction_date, name, title, issuer,
                    is_board_director, transaction_shares, transaction_price_per_share,
                    transaction_value, shares_owned_before_transaction,
                    shares_owned_after_transaction, security_title
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.ticker,
                trade.filing_date,
                trade.transaction_date,
                trade.name,
                trade.title,
                trade.issuer,
                trade.is_board_director,
                trade.transaction_shares,
                trade.transaction_price_per_share,
                trade.transaction_value,
                trade.shares_owned_before_transaction,
                trade.shares_owned_after_transaction,
                trade.security_title
            ))
            
            if cursor.rowcount > 0:
                new_trades_count += 1
                
            # Track latest transaction date
            if trade.transaction_date:
                if latest_trade_date is None or trade.transaction_date > latest_trade_date:
                    latest_trade_date = trade.transaction_date
                    
        except sqlite3.Error as e:
            logger.warning(f"Failed to insert trade for {ticker}: {e}")
            continue
    
    # Always update sync status (even if no trades) to prevent repeated downloads
    cursor.execute('''
        INSERT OR REPLACE INTO ticker_sync_status (
            ticker, last_sync_date, last_trade_date, total_trades
        ) VALUES (?, ?, ?, (
            SELECT COUNT(*) FROM insider_trades WHERE ticker = ?
        ))
    ''', (ticker, datetime.now().date().isoformat(), latest_trade_date, ticker))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Saved {new_trades_count} new insider trades for {ticker}")
    return new_trades_count


def get_insider_trades(ticker: str, limit: Optional[int] = None) -> List[InsiderTrade]:
    """
    Retrieve insider trades for a specific ticker from the database.
    
    Args:
        ticker: Stock ticker symbol
        limit: Maximum number of trades to return (most recent first)
        
    Returns:
        List of InsiderTrade objects
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = '''
        SELECT ticker, issuer, name, title, is_board_director, transaction_date,
               transaction_shares, transaction_price_per_share, transaction_value,
               shares_owned_before_transaction, shares_owned_after_transaction,
               security_title, filing_date
        FROM insider_trades 
        WHERE ticker = ?
        ORDER BY transaction_date DESC, filing_date DESC
    '''
    
    if limit:
        query += f' LIMIT {limit}'
    
    cursor.execute(query, (ticker,))
    rows = cursor.fetchall()
    conn.close()
    
    trades = []
    for row in rows:
        trade = InsiderTrade(
            ticker=row[0],
            issuer=row[1],
            name=row[2],
            title=row[3],
            is_board_director=row[4],
            transaction_date=row[5],
            transaction_shares=row[6],
            transaction_price_per_share=row[7],
            transaction_value=row[8],
            shares_owned_before_transaction=row[9],
            shares_owned_after_transaction=row[10],
            security_title=row[11],
            filing_date=row[12]
        )
        trades.append(trade)
    
    return trades


def get_multiple_insider_trades(tickers: List[str]) -> dict[str, List[InsiderTrade]]:
    """
    Retrieve insider trades for multiple tickers from the database.
    
    Args:
        tickers: List of stock ticker symbols
        
    Returns:
        Dictionary mapping ticker to list of InsiderTrade objects
    """
    if not tickers:
        return {}
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create placeholders for SQL IN clause
    placeholders = ','.join('?' * len(tickers))
    
    query = f'''
        SELECT ticker, issuer, name, title, is_board_director, transaction_date,
               transaction_shares, transaction_price_per_share, transaction_value,
               shares_owned_before_transaction, shares_owned_after_transaction,
               security_title, filing_date
        FROM insider_trades 
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, transaction_date DESC, filing_date DESC
    '''
    
    cursor.execute(query, tickers)
    rows = cursor.fetchall()
    conn.close()
    
    # Group results by ticker
    results = {ticker: [] for ticker in tickers}
    
    for row in rows:
        ticker = row[0]
        trade = InsiderTrade(
            ticker=row[0],
            issuer=row[1],
            name=row[2],
            title=row[3],
            is_board_director=row[4],
            transaction_date=row[5],
            transaction_shares=row[6],
            transaction_price_per_share=row[7],
            transaction_value=row[8],
            shares_owned_before_transaction=row[9],
            shares_owned_after_transaction=row[10],
            security_title=row[11],
            filing_date=row[12]
        )
        results[ticker].append(trade)
    
    return results


def is_ticker_cache_valid(ticker: str, max_age_days: int = 1) -> bool:
    """
    Check if the cached insider trades data for a ticker is still valid.
    
    Args:
        ticker: Stock ticker symbol
        max_age_days: Maximum age in days for cache to be considered valid
        
    Returns:
        True if cache is valid, False otherwise
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT last_sync_date FROM ticker_sync_status WHERE ticker = ?
    ''', (ticker,))
    
    result = cursor.fetchone()
    conn.close()
    
    if not result or not result[0]:
        return False
    
    try:
        last_sync = datetime.fromisoformat(result[0])
        age_days = (datetime.now() - last_sync).days
        return age_days <= max_age_days
    except (ValueError, TypeError):
        return False


def get_tickers_needing_update(tickers: List[str], max_age_days: int = 30) -> List[str]:
    """
    Get list of tickers that need insider trades data updates.
    
    Args:
        tickers: List of ticker symbols to check
        max_age_days: Maximum age in days before update is needed
        
    Returns:
        List of tickers needing updates
    """
    if not tickers:
        return []
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create placeholders for SQL IN clause
    placeholders = ','.join('?' * len(tickers))
    
    cursor.execute(f'''
        SELECT ticker FROM ticker_sync_status 
        WHERE ticker IN ({placeholders})
        AND last_sync_date >= date('now', '-{max_age_days} days')
    ''', tickers)
    
    up_to_date_tickers = {row[0] for row in cursor.fetchall()}
    conn.close()
    
    # Return tickers not in up-to-date list
    return [ticker for ticker in tickers if ticker not in up_to_date_tickers]


def get_global_most_recent_trade_date() -> Optional[str]:
    """
    Get the most recent transaction date across all tickers in the database.
    
    Returns:
        Most recent transaction date as string (YYYY-MM-DD) or None if no trades exist
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT MAX(transaction_date) FROM insider_trades
        WHERE transaction_date IS NOT NULL
    ''')
    
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result and result[0] else None


def get_database_stats() -> dict:
    """
    Get statistics about the insider trades database.
    
    Returns:
        Dictionary with database statistics
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total trades count
    cursor.execute('SELECT COUNT(*) FROM insider_trades')
    total_trades = cursor.fetchone()[0]
    
    # Unique tickers count
    cursor.execute('SELECT COUNT(DISTINCT ticker) FROM insider_trades')
    unique_tickers = cursor.fetchone()[0]
    
    # Recent trades count (last 30 days)
    cursor.execute('''
        SELECT COUNT(*) FROM insider_trades 
        WHERE transaction_date >= date('now', '-30 days')
    ''')
    recent_trades = cursor.fetchone()[0]
    
    # Database file size
    db_size_bytes = DB_PATH.stat().st_size if DB_PATH.exists() else 0
    
    conn.close()
    
    return {
        "total_trades": total_trades,
        "unique_tickers": unique_tickers,
        "recent_trades_30d": recent_trades,
        "database_size_bytes": db_size_bytes,
        "database_size_mb": round(db_size_bytes / (1024 * 1024), 2)
    }


# Initialize database on module import
setup_database()