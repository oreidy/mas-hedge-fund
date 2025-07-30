from diskcache import Cache
import time
from datetime import timedelta
import os
from pathlib import Path
from typing import Optional, List
from .insider_trades_db import (
    get_insider_trades,
    save_insider_trades,
    is_ticker_cache_valid
)
from .company_news_db import (
    get_company_news as get_company_news_from_db,
    save_company_news,
    is_news_cache_valid
)
from .models import InsiderTrade

# Define project root and cache directory as absolute paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"

class DiskCache:
    """Enhanced cache with disk persistence and expiration"""
    
    def __init__(self, cache_dir=CACHE_DIR):
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Different TTLs for different data types
        self.ttls = {
            "prices": timedelta(days=60),           # Price data cached for 2 months
            "financial_metrics": timedelta(days=60), # Financial metrics cached for 2 months
            "fred_data": timedelta(days=60),         # FRED data cached for 2 months
            "sp500_membership": timedelta(days=60),  # S&P 500 membership data cached for 2 months
            "market_cap": timedelta(days=60)         # Market cap data cached for 2 months
        }
        
        # Create separate caches for different data types
        self.caches = {
            data_type: Cache(os.path.join(cache_dir, data_type))
            for data_type in self.ttls.keys()
        }
    
    
    def close(self):
        """Explicitly close all cache connections"""
        if hasattr(self, 'caches'):
            for cache in self.caches.values():
                try:
                    cache.close()
                except Exception:
                    pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup"""
        self.close()
    
    def _get_expiration(self, data_type):
        """Get expiration time in seconds for data type"""
        return int(self.ttls.get(data_type, timedelta(days=1)).total_seconds())
    
    def get_prices(self, ticker, start_date=None, end_date=None):
        """Get cached price data with optional date filtering"""
        cache_key = f"prices:{ticker}"
        cached_data = self.caches["prices"].get(cache_key)
        
        if cached_data:
            # If date filtering is required, filter the cached data
            if start_date or end_date:
                filtered_data = [
                    price for price in cached_data 
                    if (not start_date or price["time"] >= start_date) and
                       (not end_date or price["time"] <= end_date) #end_date is inclusive
                ]
                return filtered_data
            return cached_data
        return None
    
    def set_prices(self, ticker, data):
        """Cache price data with expiration"""
        cache_key = f"prices:{ticker}"
        self.caches["prices"].set(
            cache_key, 
            data, 
            expire=self._get_expiration("prices")
        )
    
    def get_financial_metrics(self, ticker, monthly_period, period):
        """Get cached financial metrics using monthly cache period"""
        cache_key = f"financial_metrics:{ticker}:{monthly_period}:{period}"
        return self.caches["financial_metrics"].get(cache_key)
    
    def set_financial_metrics(self, ticker, monthly_period, period, data):
        """Cache financial metrics with expiration using monthly cache period"""
        cache_key = f"financial_metrics:{ticker}:{monthly_period}:{period}"
        self.caches["financial_metrics"].set(
            cache_key, 
            data, 
            expire=self._get_expiration("financial_metrics")
        )
    
    def get_line_items(self, ticker, end_date, period, line_items_list):
        """Get cached line items"""
        line_items_str = ":".join(sorted(line_items_list)) 
        cache_key = f"line_items:{ticker}:{end_date}:{period}:{line_items_str}"
        return self.caches["financial_metrics"].get(cache_key)
    
    def set_line_items(self, ticker, end_date, period, line_items_list, data):
        """Cache line items with expiration"""
        line_items_str = ":".join(sorted(line_items_list))
        cache_key = f"line_items:{ticker}:{end_date}:{period}:{line_items_str}"
        self.caches["financial_metrics"].set(
            cache_key, 
            data, 
            expire=self._get_expiration("financial_metrics")
        )
    
    def get_insider_trades(self, ticker: str, limit: Optional[int] = None) -> List[InsiderTrade]:
        """Get insider trades from SQLite database"""
        return get_insider_trades(ticker, limit)
    
    def set_insider_trades(self, ticker: str, data: List[InsiderTrade]) -> int:
        """Save insider trades to SQLite database"""
        return save_insider_trades(ticker, data)
    
    def is_insider_trades_cache_valid(self, ticker: str, max_age_days: int = 1) -> bool:
        """Check if insider trades cache is still valid"""
        return is_ticker_cache_valid(ticker, max_age_days)
    
    def get_company_news(self, ticker: str):
        """Get company news from SQLite database"""
        return get_company_news_from_db(ticker)
    
    def set_company_news(self, ticker: str, data):
        """Save company news to SQLite database"""
        return save_company_news(ticker, data)
    
    def is_company_news_cache_valid(self, ticker: str, max_age_days: int = 1) -> bool:
        """Check if company news cache is still valid"""
        return is_news_cache_valid(ticker, max_age_days)
    
    def get_outstanding_shares(self, ticker: str, date: str):
        """Get cached outstanding shares count for a ticker at a specific date."""
        cache_key = f"outstanding_shares:{ticker}:{date}"
        # Use existing financial_metrics cache for outstanding shares data
        return self.caches["financial_metrics"].get(cache_key)

    def set_outstanding_shares(self, ticker: str, date: str, shares: float):
        """Cache outstanding shares count for a ticker at a specific date."""
        cache_key = f"outstanding_shares:{ticker}:{date}"
        # Use existing financial_metrics cache for outstanding shares data
        self.caches["financial_metrics"].set(
            cache_key, 
            shares, 
            expire=self._get_expiration("financial_metrics")
        )

    def get_market_cap(self, ticker: str, date: str):
        """Get cached market cap for a ticker at a specific date."""
        cache_key = f"{ticker}:{date}"
        # Use dedicated market_cap cache for daily granularity
        return self.caches["market_cap"].get(cache_key)

    def set_market_cap(self, ticker: str, date: str, market_cap: float):
        """Cache market cap for a ticker at a specific date."""
        cache_key = f"{ticker}:{date}"
        # Use dedicated market_cap cache for daily granularity
        self.caches["market_cap"].set(
            cache_key, 
            market_cap, 
            expire=self._get_expiration("market_cap")
        )

    def update_prices(self, ticker, new_data):
        """Smart update that only adds new prices"""
        cache_key = f"prices:{ticker}"
        cached_data = self.caches["prices"].get(cache_key, [])
        
        # Create a set of dates we already have
        existing_dates = {item["time"] for item in cached_data}
        
        # Only add items with new dates
        updated_data = cached_data + [
            item for item in new_data 
            if item["time"] not in existing_dates
        ]
        
        self.caches["prices"].set(
            cache_key, 
            updated_data, 
            expire=self._get_expiration("prices")
        )
        
        return updated_data

    def get_fred_data(self, series_id, start_date, end_date):
        """Get cached FRED data"""
        cache_key = f"fred_data:{series_id}:{start_date}:{end_date}"
        return self.caches["fred_data"].get(cache_key)
    
    def set_fred_data(self, series_id, data, start_date, end_date):
        """Cache FRED data with expiration using requested date range"""
        if data:
            cache_key = f"fred_data:{series_id}:{start_date}:{end_date}"
            self.caches["fred_data"].set(
                cache_key, 
                data, 
                expire=self._get_expiration("fred_data")
            )

    def get_sp500_membership_data(self, years_back=4):
        """Get cached S&P 500 membership data"""
        cache_key = f"sp500_membership:{years_back}"
        return self.caches["sp500_membership"].get(cache_key)
    
    def set_sp500_membership_data(self, data, years_back=4):
        """Cache S&P 500 membership data with expiration"""
        if data:
            cache_key = f"sp500_membership:{years_back}"
            self.caches["sp500_membership"].set(
                cache_key, 
                data, 
                expire=self._get_expiration("sp500_membership")
            )

    def clear(self, data_type=None):
        """Clear cache for specific data type or all caches"""
        if data_type:
            if data_type in self.caches:
                self.caches[data_type].clear()
        else:
            for cache in self.caches.values():
                cache.clear()