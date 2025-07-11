"""
Advanced Risk Metrics Module

This module implements Conditional Value at Risk (CVaR) / Expected Shortfall 
calculations using GARCH(1,1) models for improved volatility forecasting.

CVaR measures the expected loss in the worst-case scenarios (tail risk),
providing a more comprehensive risk assessment than simple volatility measures.
"""

import numpy as np
import pandas as pd
from typing import Dict
import warnings
import time
from utils.logger import logger

# Suppress ARCH warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning, module='arch')

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("ARCH library not available. CVaR calculations will use simplified volatility models.", module="risk_metrics")

def calculate_cvar(
    prices_df: pd.DataFrame,
    confidence_level: float = 0.05,
    verbose_data: bool = False
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) using GARCH(1,1) model.
    
    CVaR represents the expected loss in the worst-case scenarios beyond the VaR threshold.
    For 95% confidence (5% CVaR), it's the expected loss in the worst 5% of outcomes.
    
    Args:
        prices_df: DataFrame with OHLC price data
        confidence_level: Confidence level for CVaR calculation (default 0.05 for 95% confidence)
        verbose_data: Enable detailed logging
        
    Returns:
        CVaR value (negative value indicates expected loss)
    """
    if len(prices_df) < 100:
        logger.warning(f"Insufficient data for GARCH model: {len(prices_df)} observations. Minimum 100 recommended.", module="risk_metrics")
    
    # Calculate returns
    returns = prices_df['close'].pct_change().dropna()
    
    if len(returns) < 50:
        return _fallback_cvar_calculation(returns, confidence_level, verbose_data)
    
    try:
        # Use GARCH model if ARCH library is available
        if ARCH_AVAILABLE:
            return _garch_cvar_calculation(returns, confidence_level, verbose_data)
        else:
            return _fallback_cvar_calculation(returns, confidence_level, verbose_data)
            
    except Exception as e:
        logger.warning(f"GARCH CVaR calculation failed: {str(e)}. Using fallback method.", module="risk_metrics")
        return _fallback_cvar_calculation(returns, confidence_level, verbose_data)


def _garch_cvar_calculation(
    returns: pd.Series,
    confidence_level: float,
    verbose_data: bool
) -> float:
    """Calculate CVaR using GARCH(1,1) model."""
    
    # Convert returns to percentage for better numerical stability
    returns_pct = returns * 100
    
    # Time the GARCH model fitting
    garch_start_time = time.time()
    
    # Fit GARCH(1,1) model
    model = arch_model(
        returns_pct,
        vol='GARCH',
        p=1,  # Always GARCH(1,1)
        q=1,  # Always GARCH(1,1)
        dist='normal',
        rescale=False
    )
    
    # Fit the model with reduced output
    fitted_model = model.fit(disp='off', show_warning=False)
    
    garch_fit_time = time.time() - garch_start_time
    
    # Get volatility forecast for next period
    forecast = fitted_model.forecast(horizon=1)
    volatility_forecast = np.sqrt(forecast.variance.iloc[-1, 0]) / 100  # Convert back to decimal
    
    # Calculate expected return (simple historical mean)
    expected_return = returns.mean()
    
    # Calculate CVaR using the forecasted volatility
    # For normal distribution: CVaR = μ - σ * φ(Φ^(-1)(α)) / α
    # where φ is PDF, Φ^(-1) is inverse CDF, α is confidence level
    from scipy.stats import norm
    phi_var = norm.pdf(norm.ppf(confidence_level))
    cvar = expected_return - volatility_forecast * phi_var / confidence_level
    
    if verbose_data:
        logger.debug(f"GARCH CVaR Calculation Results:", module="risk_metrics")
        logger.debug(f"  GARCH fitting time: {garch_fit_time:.3f} seconds", module="risk_metrics")
        logger.debug(f"  Data points used: {len(returns)}", module="risk_metrics")
        logger.debug(f"  Expected Return: {expected_return*100:.4f}%", module="risk_metrics")
        logger.debug(f"  Volatility Forecast: {volatility_forecast*100:.4f}%", module="risk_metrics")
        logger.debug(f"  CVaR ({(1-confidence_level)*100:.0f}% confidence): {cvar*100:.4f}%", module="risk_metrics")
    
    # Always log timing for performance monitoring
    logger.debug(f"GARCH(1,1) model fitted in {garch_fit_time:.3f}s for {len(returns)} data points", module="risk_metrics")
    
    return float(cvar)


def _fallback_cvar_calculation(
    returns: pd.Series,
    confidence_level: float,
    verbose_data: bool
) -> float:
    """Fallback CVaR calculation using historical simulation when GARCH is not available."""
    
    if len(returns) < 20:
        logger.warning(f"Very limited data for CVaR calculation: {len(returns)} observations", module="risk_metrics")
    
    # Historical simulation approach
    expected_return = returns.mean()
    volatility = returns.std()
    
    # Calculate VaR using historical quantile
    var_quantile = returns.quantile(confidence_level)
    
    # Calculate CVaR as the mean of losses beyond VaR
    tail_losses = returns[returns <= var_quantile]
    if len(tail_losses) > 0:
        cvar = tail_losses.mean()
    else:
        # If no observations in tail, use parametric approach
        from scipy.stats import norm
        phi_var = norm.pdf(norm.ppf(confidence_level))
        cvar = expected_return - volatility * phi_var / confidence_level
    
    if verbose_data:
        logger.debug(f"Historical CVaR Calculation Results:", module="risk_metrics")
        logger.debug(f"  Expected Return: {expected_return*100:.4f}%", module="risk_metrics")
        logger.debug(f"  Historical Volatility: {volatility*100:.4f}%", module="risk_metrics")
        logger.debug(f"  CVaR ({(1-confidence_level)*100:.0f}% confidence): {cvar*100:.4f}%", module="risk_metrics")
    
    return float(cvar)