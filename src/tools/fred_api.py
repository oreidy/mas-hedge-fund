import os
import pandas as pd
from typing import Dict, List
from datetime import datetime
from pathlib import Path
from fredapi import Fred

from src.data.disk_cache import DiskCache
from src.utils.logger import logger

# Create a global cache instance with disk persistence
_cache = DiskCache(cache_dir=Path(__file__).parent.parent.parent / "cache")


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
                module="get_fred_data")
    
    # Check cache first
    cached_data = _cache.get_fred_data(series_id, start_date, end_date)
    if cached_data:
        logger.debug(f"Retrieved FRED data for {series_id} from cache", 
                    module="get_fred_data")
        df = pd.DataFrame(cached_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    
    # Get FRED API key
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.error("FRED_API_KEY not found in environment variables", 
                    module="get_fred_data")
        return pd.DataFrame()
    
    try:
        # Initialize FRED client
        fred = Fred(api_key=api_key)
        
        logger.debug(f"Downloading FRED series: {series_id}", 
                    module="get_fred_data")
        
        # Determine units transformation based on series
        if series_id in ['CPIAUCSL', 'PCEPI']:
            # Get 12-month percentage change for inflation data
            units = 'pc1'
        else:
            # Use levels for other series (Fed Funds, GDP already comes as % change)
            units = 'lin'
        
        # Fetch data with appropriate transformation
        data = fred.get_series(series_id, start=start_date, end=end_date, units=units)
        
        if data.empty:
            logger.warning(f"No data found for {series_id}", 
                         module="get_fred_data")
            return pd.DataFrame()
        
        # DEBUG: Log what date range we actually got from FRED API
        if verbose_data:
            logger.debug(f"FRED API returned data for {series_id}: {len(data)} observations", 
                        module="get_fred_data")
            logger.debug(f"FRED data date range: {data.index.min()} to {data.index.max()}", 
                        module="get_fred_data")
            logger.debug(f"Requested date range: {start_date} to {end_date}", 
                        module="get_fred_data")
        
        # Convert to DataFrame with proper format
        df = data.to_frame(name='value')
        df.index.name = 'date'
        
        # Filter data by date range since FRED API ignores date parameters
        start_date_parsed = pd.to_datetime(start_date)
        end_date_parsed = pd.to_datetime(end_date)
        df_filtered = df[(df.index >= start_date_parsed) & (df.index <= end_date_parsed)]
        
        if verbose_data:
            logger.debug(f"FRED data filtered from {len(df)} to {len(df_filtered)} observations", 
                        module="get_fred_data")
            logger.debug(f"Filtered date range: {df_filtered.index.min()} to {df_filtered.index.max()}", 
                        module="get_fred_data")
        
        df = df_filtered.reset_index()
        
        # Cache the results
        cache_data = [{'date': row['date'].strftime('%Y-%m-%d'), 'value': row['value'], 'series_id': series_id} 
                     for _, row in df.iterrows() if pd.notna(row['value'])]
        
        if cache_data:
            _cache.set_fred_data(series_id, cache_data, start_date, end_date)
        
        # Return DataFrame with date index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        if verbose_data:
            logger.debug(f"Fetched {len(df)} observations for {series_id}", 
                        module="get_fred_data")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching FRED series {series_id}: {e}", 
                    module="get_fred_data")
        return pd.DataFrame()


def get_vix_data(start_date: str, end_date: str, verbose_data: bool = False) -> pd.DataFrame:
    """
    Fetch VIX (Volatility Index) data from FRED API.
    The VIX measures implied volatility and market uncertainty.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        verbose_data: Optional flag for verbose logging
        
    Returns:
        DataFrame with VIX values
    """
    
    logger.debug(f"Fetching VIX data from {start_date} to {end_date}", 
                module="get_vix_data")
    
    return get_fred_data('VIXCLS', start_date, end_date, verbose_data)


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
    
    # Key indicators from research proposal - using 12-month percentage change series
    indicators = {
        'cpi': 'CPIAUCSL',      # Consumer Price Index (12-month % change)
        'pce': 'PCEPI',         # Personal Consumption Expenditures Price Index (12-month % change)
        'gdp': 'A191RL1Q225SBEA', # Real GDP (% change from year ago, quarterly)
        'fed_funds': 'FEDFUNDS', # Federal Funds Rate (level)
    }
    
    logger.debug(f"Fetching macro indicators for allocation", 
                module="get_macro_indicators_for_allocation")
    
    results = {}
    for name, series_id in indicators.items():
        results[name] = get_fred_data(series_id, start_date, end_date, verbose_data)
    
    return results


def get_all_fred_data_for_agents(start_date: str, end_date: str, verbose_data: bool = False) -> None:
    """
    Prefetch all FRED data needed by macro, forward-looking, and fixed-income agents.
    Similar to get_data_for_tickers but for FRED macroeconomic data.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        verbose_data: Optional flag for verbose logging
        
    Returns:
        None - Data is cached for later use by agents
    """
    
    logger.debug(f"Prefetching all FRED data from {start_date} to {end_date}", 
                module="get_all_fred_data_for_agents")
    
    try:
        # Prefetch macro indicators (for macro agent)
        macro_indicators = get_macro_indicators_for_allocation(start_date, end_date, verbose_data)
        
        # Prefetch VIX data (for forward-looking agent)
        vix_data = get_vix_data(start_date, end_date, verbose_data)
        
        # Prefetch treasury yield data (for fixed-income agent)
        treasury_2y = get_fred_data("GS2", start_date, end_date, verbose_data)
        treasury_10y = get_fred_data("GS10", start_date, end_date, verbose_data)
        
        if verbose_data:
            logger.debug("FRED data prefetching summary:", module="get_all_fred_data_for_agents")
            logger.debug(f"  - Macro indicators: {len(macro_indicators)} series", module="get_all_fred_data_for_agents")
            logger.debug(f"  - VIX data: {len(vix_data)} observations", module="get_all_fred_data_for_agents")
            logger.debug(f"  - Treasury 2Y: {len(treasury_2y)} observations", module="get_all_fred_data_for_agents")
            logger.debug(f"  - Treasury 10Y: {len(treasury_10y)} observations", module="get_all_fred_data_for_agents")
        
    except Exception as e:
        logger.error(f"Error prefetching FRED data: {e}", module="get_all_fred_data_for_agents")
        raise