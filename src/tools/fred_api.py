import os
import pandas as pd
from typing import Dict, List
from datetime import datetime
from pathlib import Path
from fredapi import Fred

from data.disk_cache import DiskCache
from utils.logger import logger

# Create a global cache instance with disk persistence
_cache = DiskCache(cache_dir=Path("./cache"))


def get_fred_data(series_id: str, start_date: str, end_date: str, verbose_data: bool = False) -> pd.DataFrame:
    """
    Fetch macroeconomic data from FRED API for a single series.
    
    Args:
        series_id: FRED series identifier (e.g., 'GDP', 'UNRATE', 'FEDFUNDS')
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        verbose_data: Optional flag for verbose logging
        
    Returns:
        DataFrame with date index and values
    """
    
    logger.debug(f"Fetching FRED data for {series_id} from {start_date} to {end_date}", 
                module="get_fred_data", series_id=series_id)
    
    # Check cache first
    cached_data = _cache.get_fred_data(series_id, start_date, end_date)
    if cached_data:
        logger.debug(f"Retrieved FRED data for {series_id} from cache", 
                    module="get_fred_data", series_id=series_id)
        df = pd.DataFrame(cached_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    
    # Get FRED API key
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.error("FRED_API_KEY not found in environment variables", 
                    module="get_fred_data", series_id=series_id)
        return pd.DataFrame()
    
    try:
        # Initialize FRED client
        fred = Fred(api_key=api_key)
        
        logger.debug(f"Downloading FRED series: {series_id}", 
                    module="get_fred_data", series_id=series_id)
        
        # Fetch data
        data = fred.get_series(series_id, start=start_date, end=end_date)
        
        if data.empty:
            logger.warning(f"No data found for {series_id}", 
                         module="get_fred_data", series_id=series_id)
            return pd.DataFrame()
        
        # Convert to DataFrame with proper format
        df = data.to_frame(name='value')
        df.index.name = 'date'
        df = df.reset_index()
        
        # Cache the results
        cache_data = [{'date': row['date'].strftime('%Y-%m-%d'), 'value': row['value'], 'series_id': series_id} 
                     for _, row in df.iterrows() if pd.notna(row['value'])]
        
        if cache_data:
            _cache.set_fred_data(series_id, cache_data)
        
        # Return DataFrame with date index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        if verbose_data:
            logger.debug(f"Fetched {len(df)} observations for {series_id}", 
                        module="get_fred_data", series_id=series_id)
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching FRED series {series_id}: {e}", 
                    module="get_fred_data", series_id=series_id)
        return pd.DataFrame()


def get_macro_indicators_for_allocation(start_date: str, end_date: str, verbose_data: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Get key macro indicators for asset allocation decisions.
    Based on research proposal: CPI, PCE, GDP growth, Federal Funds Rate, VIX
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        verbose_data: Optional flag for verbose logging
        
    Returns:
        Dictionary with macro indicators for allocation decisions
    """
    
    # Key indicators from research proposal
    indicators = {
        'cpi': 'CPIAUCSL',      # Consumer Price Index
        'pce': 'PCEPI',         # Personal Consumption Expenditures Price Index
        'gdp': 'GDP',           # Gross Domestic Product
        'fed_funds': 'FEDFUNDS', # Federal Funds Rate
        'vix': 'VIXCLS',        # VIX (volatility index)
    }
    
    logger.debug(f"Fetching macro indicators for allocation", 
                module="get_macro_indicators_for_allocation")
    
    results = {}
    for name, series_id in indicators.items():
        results[name] = get_fred_data(series_id, start_date, end_date, verbose_data)
    
    return results