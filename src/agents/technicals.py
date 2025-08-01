import math

from langchain_core.messages import HumanMessage

from graph.state import AgentState, show_agent_reasoning

import json
import pandas as pd
import numpy as np
from math import log

from tools.api import get_price_data, prices_to_df
from utils.progress import progress
from utils.logger import logger



##### Technical Analyst #####
def technical_analyst_agent(state: AgentState):
    """
    Sophisticated technical analysis system that combines multiple trading strategies for multiple tickers:
    1. Trend Following
    2. Mean Reversion
    3. Momentum
    4. Volatility Analysis
    5. Statistical Arbitrage Signals
    """

    # Get verbose_data from metadata or default to False
    verbose_data = state["metadata"].get("verbose_data", False)
    logger.debug("Accessing Technicals Agent", module="technicals_agent")

    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize analysis for each ticker
    technical_analysis = {}

    for ticker in tickers:
        progress.update_status("technical_analyst_agent", ticker, "Analyzing price data")

        # Get the historical price data
        logger.debug(f"Requesting price data for {ticker} from {start_date} to {end_date}", 
                    module="technical_analyst_agent", ticker=ticker)
        
        prices_df = get_price_data(ticker=ticker, 
                                   start_date=start_date, 
                                   end_date=end_date,
                                   verbose_data=verbose_data)
        
        logger.debug(f"Received {len(prices_df)} days of price data for {ticker}", 
                    module="technical_analyst_agent", ticker=ticker)
        
        if verbose_data:
            logger.debug(f"prices_df: {prices_df}", module="technical_analyst_agent")

        if prices_df.empty:
            progress.update_status("technical_analyst_agent", ticker, "Failed: No price data found")
            continue

        if len(prices_df) < 127: # For 6-month momentum 126 returns are needed.
            logger.warning(
                f"Insufficient historical data: {len(prices_df)} days available, "
                f"Some indicators may return NaN values or default signals.",
                module="technical_analyst_agent",
                ticker=ticker
            )

        progress.update_status("technical_analyst_agent", ticker, "Calculating trend signals")
        trend_signals = calculate_trend_signals(prices_df, ticker, verbose_data)

        progress.update_status("technical_analyst_agent", ticker, "Calculating mean reversion")
        mean_reversion_signals = calculate_mean_reversion_signals(prices_df, ticker, verbose_data)

        progress.update_status("technical_analyst_agent", ticker, "Calculating momentum")
        momentum_signals = calculate_momentum_signals(prices_df, ticker, verbose_data)

        progress.update_status("technical_analyst_agent", ticker, "Analyzing volatility")
        volatility_signals = calculate_volatility_signals(prices_df, ticker, verbose_data)

        progress.update_status("technical_analyst_agent", ticker, "Statistical analysis")
        stat_arb_signals = calculate_stat_arb_signals(prices_df, ticker, verbose_data)

        # Combine all signals using a weighted ensemble approach
        strategy_weights = {
            "trend": 0.25,
            "mean_reversion": 0.20,
            "momentum": 0.25,
            "volatility": 0.15,
            "stat_arb": 0.15,
        }

        progress.update_status("technical_analyst_agent", ticker, "Combining signals")
        combined_signal = weighted_signal_combination(
            {
                "trend": trend_signals,
                "mean_reversion": mean_reversion_signals,
                "momentum": momentum_signals,
                "volatility": volatility_signals,
                "stat_arb": stat_arb_signals,
            },
            strategy_weights,
        )
        if verbose_data:
            logger.debug(
                f"=== Technical Analysis Summary for {ticker} ===\n"
                f"Trend Signal: {trend_signals['signal']} (Confidence: {trend_signals['confidence']:.2f})\n"
                f"Mean Reversion Signal: {mean_reversion_signals['signal']} (Confidence: {mean_reversion_signals['confidence']:.2f})\n"
                f"Momentum Signal: {momentum_signals['signal']} (Confidence: {momentum_signals['confidence']:.2f})\n"
                f"Volatility Signal: {volatility_signals['signal']} (Confidence: {volatility_signals['confidence']:.2f})\n"
                f"Stat Arb Signal: {stat_arb_signals['signal']} (Confidence: {stat_arb_signals['confidence']:.2f})\n"
                f"COMBINED Signal: {combined_signal['signal']} (Confidence: {combined_signal['confidence']:.2f})",
                module="technical_analyst_agent", 
                ticker=ticker
            )

        # Generate detailed analysis report for this ticker
        technical_analysis[ticker] = {
            "signal": combined_signal["signal"],
            "confidence": round(combined_signal["confidence"] * 100),
            "strategy_signals": {
                "trend_following": {
                    "signal": trend_signals["signal"],
                    "confidence": round(trend_signals["confidence"] * 100),
                    "metrics": normalize_pandas(trend_signals["metrics"]),
                },
                "mean_reversion": {
                    "signal": mean_reversion_signals["signal"],
                    "confidence": round(mean_reversion_signals["confidence"] * 100),
                    "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                },
                "momentum": {
                    "signal": momentum_signals["signal"],
                    "confidence": round(momentum_signals["confidence"] * 100),
                    "metrics": normalize_pandas(momentum_signals["metrics"]),
                },
                "volatility": {
                    "signal": volatility_signals["signal"],
                    "confidence": round(volatility_signals["confidence"] * 100),
                    "metrics": normalize_pandas(volatility_signals["metrics"]),
                },
                "statistical_arbitrage": {
                    "signal": stat_arb_signals["signal"],
                    "confidence": round(stat_arb_signals["confidence"] * 100),
                    "metrics": normalize_pandas(stat_arb_signals["metrics"]),
                },
            },
        }
        progress.update_status("technical_analyst_agent", ticker, "Done")

    # Create the technical analyst message
    message = HumanMessage(
        content=json.dumps(technical_analysis),
        name="technical_analyst_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(technical_analysis, "Technical Analyst")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def calculate_trend_signals(prices_df, ticker, verbose_data):
    """
    Advanced trend following strategy using multiple timeframes and indicators
    """
    # Calculate EMAs for multiple timeframes
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)

    # Calculate ADX for trend strength (Average Directional Index)
    adx = calculate_adx(prices_df, 14)

    # Determine trend direction and strength
    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55

    # Combine signals with confidence weighting
    trend_strength = adx["adx"].iloc[-1] / 100.0

    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = "bullish"
        confidence = trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = "bearish"
        confidence = trend_strength
    else:
        signal = "neutral"
        confidence = 0.5

    if verbose_data:
        logger.debug(
            f"TREND ANALYSIS: Signal={signal}, Confidence={confidence:.2f}\n"
            f"- EMA: 8={ema_8.iloc[-1]:.2f}, 21={ema_21.iloc[-1]:.2f}, 55={ema_55.iloc[-1]:.2f}\n"
            f"- Trends: Short={'Bullish' if short_trend.iloc[-1] else 'Bearish'}, Medium={'Bullish' if medium_trend.iloc[-1] else 'Bearish'}\n"
            f"- ADX: {adx['adx'].iloc[-1]:.2f}, Trend Strength: {trend_strength:.2f}",
            module="calculate_trend_signals",
            ticker=ticker
        )
    
    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": float(adx["adx"].iloc[-1]),
            "trend_strength": float(trend_strength),
        },
    }


def calculate_mean_reversion_signals(prices_df, ticker, verbose_data):
    """
    Mean reversion strategy using statistical measures and Bollinger Bands
    """
    # Calculate z-score of price relative to moving average
    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()
    z_score = (prices_df["close"] - ma_50) / std_50

    # Calculate Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)

    # Calculate RSI with multiple timeframes
    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)

    # Mean reversion signals
    price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    # Combine signals
    if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
        signal = "bullish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
        signal = "bearish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    if verbose_data:
        logger.debug(
            f"MEAN REVERSION: Signal={signal}, Confidence={confidence:.2f}\n"
            f"- Z-Score: {z_score.iloc[-1]:.2f}\n"
            f"- Price vs BB: {price_vs_bb:.2f} (0=at lower band, 1=at upper band)\n"
            f"- RSI: 14-day={rsi_14.iloc[-1]:.2f}, 28-day={rsi_28.iloc[-1]:.2f}",
            module="calculate_mean_reversion_signals",
            ticker=ticker
        )

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": float(z_score.iloc[-1]),
            "price_vs_bb": float(price_vs_bb),
            "rsi_14": float(rsi_14.iloc[-1]),
            "rsi_28": float(rsi_28.iloc[-1]),
        },
    }


def calculate_momentum_signals(prices_df, ticker, verbose_data):
    """
    Multi-factor momentum strategy
    """
    # Price momentum
    returns = prices_df["close"].pct_change()
    mom_1m = returns.rolling(21).sum()
    mom_3m = returns.rolling(63).sum()
    mom_6m = returns.rolling(126).sum()

    # Volume momentum
    volume_ma = prices_df["volume"].rolling(21).mean()
    volume_momentum = prices_df["volume"] / volume_ma

    # Relative strength
    # (would compare to market/sector in real implementation)

    # Calculate momentum score
    momentum_score = (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m).iloc[-1]

    # Volume confirmation
    volume_confirmation = volume_momentum.iloc[-1] > 1.0

    if momentum_score > 0.05 and volume_confirmation:
        signal = "bullish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    elif momentum_score < -0.05 and volume_confirmation:
        signal = "bearish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    if verbose_data:
        logger.debug(
            f"MOMENTUM: Signal={signal}, Confidence={confidence:.2f}\n"
            f"- Momentum Score: {momentum_score:.4f}\n"
            f"- Time Periods: 1M={mom_1m.iloc[-1]:.4f}, 3M={mom_3m.iloc[-1]:.4f}, 6M={mom_6m.iloc[-1]:.4f}\n"
            f"- Volume Confirmation: {volume_confirmation} (Vol/MA: {volume_momentum.iloc[-1]:.2f})",
            module="calculate_momentum_signals",
            ticker=ticker
        )

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_1m": float(mom_1m.iloc[-1]),
            "momentum_3m": float(mom_3m.iloc[-1]),
            "momentum_6m": float(mom_6m.iloc[-1]),
            "volume_momentum": float(volume_momentum.iloc[-1]),
        },
    }


def calculate_volatility_signals(prices_df, ticker, verbose_data):
    """
    Volatility-based trading strategy
    """
    # Calculate various volatility metrics
    returns = prices_df["close"].pct_change()

    # Historical volatility
    hist_vol = returns.rolling(21).std() * math.sqrt(252)

    # Volatility regime detection
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma

    # Volatility mean reversion
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

    # ATR ratio
    atr = calculate_atr(prices_df)
    atr_ratio = atr / prices_df["close"]

    # Generate signal based on volatility regime
    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]

    if current_vol_regime < 0.8 and vol_z < -1:
        signal = "bullish"  # Low vol regime, potential for expansion
        confidence = min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > 1.2 and vol_z > 1:
        signal = "bearish"  # High vol regime, potential for contraction
        confidence = min(abs(vol_z) / 3, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    if verbose_data:
        logger.debug(
            f"VOLATILITY: Signal={signal}, Confidence={confidence:.2f}\n"
            f"- Historical Volatility: {hist_vol.iloc[-1]:.4f} (annualized)\n"
            f"- Volatility Regime: {current_vol_regime:.2f} (<1=low, >1=high)\n"
            f"- Vol Z-Score: {vol_z:.2f}\n"
            f"- ATR Ratio: {atr_ratio.iloc[-1]:.4f}",
            module="calculate_volatility_signals",
            ticker=ticker
        )

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": float(hist_vol.iloc[-1]),
            "volatility_regime": float(current_vol_regime),
            "volatility_z_score": float(vol_z),
            "atr_ratio": float(atr_ratio.iloc[-1]),
        },
    }


def calculate_stat_arb_signals(prices_df, ticker, verbose_data):
    """
    Statistical arbitrage signals based on price action analysis
    """
    # Calculate price distribution statistics
    returns = prices_df["close"].pct_change()

    # Skewness and kurtosis
    skew = returns.rolling(63).skew()
    kurt = returns.rolling(63).kurt()

    # Test for mean reversion using Hurst exponent
    hurst = calculate_hurst_exponent(prices_df["close"])

    # Correlation analysis
    # (would include correlation with related securities in real implementation)

    # Generate signal based on statistical properties
    if hurst < 0.4 and skew.iloc[-1] > 1:
        signal = "bullish"
        confidence = (0.5 - hurst) * 2
    elif hurst < 0.4 and skew.iloc[-1] < -1:
        signal = "bearish"
        confidence = (0.5 - hurst) * 2
    else:
        signal = "neutral"
        confidence = 0.5


    if verbose_data:
        logger.debug(
            f"STATISTICAL ARB: Signal={signal}, Confidence={confidence:.2f}\n"
            f"- Hurst Exponent: {hurst:.2f} (<0.5=mean reverting, >0.5=trending)\n"
            f"- Skewness: {skew.iloc[-1]:.2f} (>0=right skew, <0=left skew)\n"
            f"- Kurtosis: {kurt.iloc[-1]:.2f} (>3=fat tails)",
            module="calculate_stat_arb_signals",
            ticker=ticker
        )

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "hurst_exponent": float(hurst),
            "skewness": float(skew.iloc[-1]),
            "kurtosis": float(kurt.iloc[-1]),
        },
    }


def weighted_signal_combination(signals, weights):
    """
    Combines multiple trading signals using a weighted approach
    """
    # Convert signals to numeric values
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}

    weighted_sum = 0
    total_confidence = 0

    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal["signal"]]
        weight = weights[strategy]
        confidence = signal["confidence"]

        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence

    # Normalize the weighted sum
    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0

    # Convert back to signal
    if final_score > 0.2:
        signal = "bullish"
    elif final_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    return {"signal": signal, "confidence": abs(final_score)}


def normalize_pandas(obj):
    """Convert pandas Series/DataFrames to primitive Python types"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average

    Args:
        df: DataFrame with price data
        window: EMA period

    Returns:
        pd.Series: EMA values
    """
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)

    Args:
        df: DataFrame with OHLC data
        period: Period for calculations

    Returns:
        DataFrame with ADX values
    """
    # Calculate True Range
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

    # Calculate Directional Movement
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]

    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)

    # Calculate ADX
    df["+di"] = 100 * (df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["-di"] = 100 * (df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())

    # Handle division by zero case
    di_sum = df["+di"] + df["-di"]
    df["dx"] = np.where(di_sum != 0, 
                       100 * abs(df["+di"] - df["-di"]) / di_sum, 
                       0)  # Set to 0 when no directional movement
    
    df["adx"] = df["dx"].ewm(span=period).mean()

    return df[["adx", "+di", "-di"]]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range

    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation

    Returns:
        pd.Series: ATR values
    """
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(period).mean()


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent to determine long-term memory of time series
    H < 0.5: Mean reverting series
    H = 0.5: Random walk
    H > 0.5: Trending series

    Args:
        price_series: Array-like price data
        max_lag: Maximum lag for R/S calculation

    Returns:
        float: Hurst exponent
    """
    # Convert price series to returns
    returns = price_series.pct_change().dropna()
    
    if len(returns) < max_lag * 2:
        return 0.5  # Not enough data
    
    # Calculate R/S for different lag values
    lags = range(2, min(max_lag, len(returns) // 2))
    rs_values = []
    
    for lag in lags:
        # Split returns into chunks of size 'lag'
        chunks = len(returns) // lag
        if chunks < 1:
            continue
            
        # Calculate R/S for each chunk and average
        rs_list = []
        for i in range(chunks):
            chunk = returns[i * lag:(i + 1) * lag]
            if len(chunk) < 2:
                continue
                
            # Calculate cumulative deviation from mean
            mean = chunk.mean()
            deviation = chunk - mean
            cumulative = deviation.cumsum()
            
            # Calculate range and standard deviation
            r = cumulative.max() - cumulative.min()
            s = chunk.std()
            
            if s > 0:
                rs_list.append(r / s)
        
        if rs_list:
            rs_values.append(np.mean(rs_list))
    
    if len(rs_values) < 4:
        return 0.5  # Not enough valid data points
    
    # Fit a line to log-log plot of R/S vs lag
    try:
        log_lags = np.log10([lags[i] for i in range(len(rs_values))])
        log_rs = np.log10(rs_values)
        reg = np.polyfit(log_lags, log_rs, 1)
        return reg[0]  # Slope is the Hurst exponent
    except Exception:
        return 0.5
