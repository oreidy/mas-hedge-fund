from diskcache import Cache
import time
from datetime import timedelta
import os
from pathlib import Path

class DiskCache:
    """Enhanced cache with disk persistence and expiration"""
    
    def __init__(self, cache_dir=Path("./cache")):
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Different TTLs for different data types
        self.ttls = {
            "prices": timedelta(hours=1),          # Price data refreshes hourly
            "financial_metrics": timedelta(days=7), # Financial metrics refresh weekly
            "insider_trades": timedelta(days=1),    # Insider trade data daily
            "company_news": timedelta(hours=4)      # News refreshes 4 times daily
        }
        
        # Create separate caches for different data types
        self.caches = {
            data_type: Cache(os.path.join(cache_dir, data_type))
            for data_type in self.ttls.keys()
        }
    
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
    
    def get_financial_metrics(self, ticker):
        """Get cached financial metrics"""
        cache_key = f"financial_metrics:{ticker}"
        return self.caches["financial_metrics"].get(cache_key)
    
    def set_financial_metrics(self, ticker, data):
        """Cache financial metrics with expiration"""
        cache_key = f"financial_metrics:{ticker}"
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
    
    def get_insider_trades(self, ticker):
        """Get cached insider trades"""
        cache_key = f"insider_trades:{ticker}"
        return self.caches["insider_trades"].get(cache_key)
    
    def set_insider_trades(self, ticker, data):
        """Cache insider trades with expiration"""
        cache_key = f"insider_trades:{ticker}"
        self.caches["insider_trades"].set(
            cache_key, 
            data, 
            expire=self._get_expiration("insider_trades")
        )
    
    def get_company_news(self, ticker):
        """Get cached company news"""
        cache_key = f"company_news:{ticker}"
        return self.caches["company_news"].get(cache_key)
    
    def set_company_news(self, ticker, data):
        """Cache company news with expiration"""
        cache_key = f"company_news:{ticker}"
        self.caches["company_news"].set(
            cache_key, 
            data, 
            expire=self._get_expiration("company_news")
        )
    
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

    def clear(self, data_type=None):
        """Clear cache for specific data type or all caches"""
        if data_type:
            if data_type in self.caches:
                self.caches[data_type].clear()
        else:
            for cache in self.caches.values():
                cache.clear()