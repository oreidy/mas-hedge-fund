"""
Centralized database utility module for company news data.
Handles all SQLite operations for company news storage and retrieval.
"""

import sqlite3
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime, date

from .models import CompanyNews

logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = Path(__file__).parent.parent.parent / "company_news_database.db"


def setup_database():
    """Initialize the company news database with required tables and indexes."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS company_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            title TEXT NOT NULL,
            author TEXT,
            source TEXT,
            date DATE,
            url TEXT,
            sentiment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, title, date, source)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS news_sync_status (
            ticker TEXT PRIMARY KEY,
            last_sync_date DATE,
            last_news_date DATE,
            total_news INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_news_ticker ON company_news(ticker)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_news_date ON company_news(date)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_news_sentiment ON company_news(sentiment)
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Company news database setup completed")


def save_company_news(ticker: str, news_list: List[CompanyNews]) -> int:
    """
    Save company news to the database and update sync status.
    
    Args:
        ticker: Stock ticker symbol
        news_list: List of CompanyNews objects to save
        
    Returns:
        Number of new news items added to database
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    new_news_count = 0
    latest_news_date = None
    
    # Get initial count for this ticker
    cursor.execute('SELECT COUNT(*) FROM company_news WHERE ticker = ?', (ticker,))
    initial_count = cursor.fetchone()[0]
    
    # Process news if any exist
    for news in news_list:
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO company_news (
                    ticker, title, author, source, date, url, sentiment
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                news.ticker,
                news.title,
                news.author,
                news.source,
                news.date,
                news.url,
                news.sentiment
            ))
            
            # Note: rowcount with INSERT OR IGNORE can be unreliable
            # We'll calculate the actual count difference at the end
                
            # Track latest news date
            if news.date:
                if latest_news_date is None or news.date > latest_news_date:
                    latest_news_date = news.date
                    
        except sqlite3.Error as e:
            logger.warning(f"Failed to insert news for {ticker}: {e}")
            continue
    
    # Calculate actual new items added by comparing counts
    cursor.execute('SELECT COUNT(*) FROM company_news WHERE ticker = ?', (ticker,))
    final_count = cursor.fetchone()[0]
    new_news_count = final_count - initial_count
    
    # Always update sync status (even if no news) to prevent repeated downloads
    cursor.execute('''
        INSERT OR REPLACE INTO news_sync_status (
            ticker, last_sync_date, last_news_date, total_news
        ) VALUES (?, ?, ?, ?)
    ''', (ticker, datetime.now().date().isoformat(), latest_news_date, final_count))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Saved {new_news_count} new company news items for {ticker}")
    return new_news_count


def get_company_news(ticker: str, limit: Optional[int] = None) -> List[CompanyNews]:
    """
    Retrieve company news for a specific ticker from the database.
    
    Args:
        ticker: Stock ticker symbol
        limit: Maximum number of news items to return (most recent first)
        
    Returns:
        List of CompanyNews objects
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = '''
        SELECT ticker, title, author, source, date, url, sentiment
        FROM company_news 
        WHERE ticker = ?
        ORDER BY date DESC, created_at DESC
    '''
    
    if limit:
        query += f' LIMIT {limit}'
    
    cursor.execute(query, (ticker,))
    rows = cursor.fetchall()
    conn.close()
    
    news_list = []
    for row in rows:
        news = CompanyNews(
            ticker=row[0],
            title=row[1],
            author=row[2],
            source=row[3],
            date=row[4],
            url=row[5],
            sentiment=row[6]
        )
        news_list.append(news)
    
    return news_list


def get_multiple_company_news(tickers: List[str]) -> dict[str, List[CompanyNews]]:
    """
    Retrieve company news for multiple tickers from the database.
    
    Args:
        tickers: List of stock ticker symbols
        
    Returns:
        Dictionary mapping ticker to list of CompanyNews objects
    """
    if not tickers:
        return {}
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create placeholders for SQL IN clause
    placeholders = ','.join('?' * len(tickers))
    
    query = f'''
        SELECT ticker, title, author, source, date, url, sentiment
        FROM company_news 
        WHERE ticker IN ({placeholders})
        ORDER BY ticker, date DESC, created_at DESC
    '''
    
    cursor.execute(query, tickers)
    rows = cursor.fetchall()
    conn.close()
    
    # Group results by ticker
    results = {ticker: [] for ticker in tickers}
    
    for row in rows:
        ticker = row[0]
        news = CompanyNews(
            ticker=row[0],
            title=row[1],
            author=row[2],
            source=row[3],
            date=row[4],
            url=row[5],
            sentiment=row[6]
        )
        results[ticker].append(news)
    
    return results


def is_news_cache_valid(ticker: str, max_age_days: int = 1) -> bool:
    """
    Check if the cached company news data for a ticker is still valid.
    
    Args:
        ticker: Stock ticker symbol
        max_age_days: Maximum age in days for cache to be considered valid
        
    Returns:
        True if cache is valid, False otherwise
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT last_sync_date FROM news_sync_status WHERE ticker = ?
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


def get_news_tickers_needing_update(tickers: List[str], max_age_days: int = 30) -> List[str]:
    """
    Get list of tickers that need company news data updates.
    
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
        SELECT ticker FROM news_sync_status 
        WHERE ticker IN ({placeholders})
        AND last_sync_date >= date('now', '-{max_age_days} days')
    ''', tickers)
    
    up_to_date_tickers = {row[0] for row in cursor.fetchall()}
    conn.close()
    
    # Return tickers not in up-to-date list
    return [ticker for ticker in tickers if ticker not in up_to_date_tickers]


def get_global_most_recent_news_date() -> Optional[str]:
    """
    Get the most recent news date across all tickers in the database.
    
    Returns:
        Most recent news date as string (YYYY-MM-DD) or None if no news exists
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT MAX(date) FROM company_news
        WHERE date IS NOT NULL
    ''')
    
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result and result[0] else None


def get_news_database_stats() -> dict:
    """
    Get statistics about the company news database.
    
    Returns:
        Dictionary with database statistics
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total news count
    cursor.execute('SELECT COUNT(*) FROM company_news')
    total_news = cursor.fetchone()[0]
    
    # Unique tickers count
    cursor.execute('SELECT COUNT(DISTINCT ticker) FROM company_news')
    unique_tickers = cursor.fetchone()[0]
    
    # Recent news count (last 30 days)
    cursor.execute('''
        SELECT COUNT(*) FROM company_news 
        WHERE date >= date('now', '-30 days')
    ''')
    recent_news = cursor.fetchone()[0]
    
    # News by sentiment
    cursor.execute('''
        SELECT sentiment, COUNT(*) FROM company_news 
        WHERE sentiment IS NOT NULL 
        GROUP BY sentiment
    ''')
    sentiment_counts = dict(cursor.fetchall())
    
    # Database file size
    db_size_bytes = DB_PATH.stat().st_size if DB_PATH.exists() else 0
    
    conn.close()
    
    return {
        "total_news": total_news,
        "unique_tickers": unique_tickers,
        "recent_news_30d": recent_news,
        "sentiment_breakdown": sentiment_counts,
        "database_size_bytes": db_size_bytes,
        "database_size_mb": round(db_size_bytes / (1024 * 1024), 2)
    }


# Initialize database on module import
setup_database()