#!/usr/bin/env python3
"""Debug script to analyze portfolio values and returns."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

from rl.backtesting.lstm_backtest import LSTMWalkForwardBacktester
import logging

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce log noise
    format='%(levelname)s:%(name)s:%(message)s'
)

def analyze_portfolio_trajectory():
    """Analyze the portfolio trajectory to find extreme values."""
    
    # Initialize backtester
    backtester = LSTMWalkForwardBacktester(
        initial_capital=100000.0,
        initial_training_episodes=250,
        max_positions=30,
        transaction_cost=0.0005,
        sequence_length=20
    )
    
    # Load episodes
    episodes = backtester.load_all_episodes("training_data/episodes")
    print(f"Loaded {len(episodes)} episodes")
    
    if len(episodes) < 300:
        print("Not enough episodes for analysis")
        return
    
    # Run backtest (but limit to first 300 episodes for speed)
    limited_episodes = episodes[:300]  # Just for debugging
    results = backtester.run_walk_forward_backtest(limited_episodes)
    
    # Analyze the daily results
    daily_results = results['daily_results']
    
    # Create DataFrame for analysis
    df = pd.DataFrame([
        {
            'date': r['date'],
            'episode_index': r['episode_index'],
            'portfolio_value_before': r['portfolio_value_before'],
            'portfolio_value_after': r['portfolio_value_after'],
            'daily_return': r['daily_return']
        }
        for r in daily_results
    ])
    
    print("\nPortfolio Analysis:")
    print(f"Initial capital: $100,000")
    print(f"Final portfolio value: ${df['portfolio_value_after'].iloc[-1]:,.2f}")
    print(f"Total return: {(df['portfolio_value_after'].iloc[-1] - 100000) / 100000:.2%}")
    
    print(f"\nDaily returns statistics:")
    print(f"Mean daily return: {df['daily_return'].mean():.6f} ({df['daily_return'].mean()*100:.4f}%)")
    print(f"Std daily return: {df['daily_return'].std():.6f}")
    print(f"Min daily return: {df['daily_return'].min():.6f} ({df['daily_return'].min()*100:.2f}%)")
    print(f"Max daily return: {df['daily_return'].max():.6f} ({df['daily_return'].max()*100:.2f}%)")
    
    # Find extreme returns
    extreme_negative = df[df['daily_return'] < -0.5]  # Less than -50%
    extreme_positive = df[df['daily_return'] > 0.5]   # More than +50%
    
    print(f"\nExtreme returns:")
    print(f"Days with return < -50%: {len(extreme_negative)}")
    print(f"Days with return > +50%: {len(extreme_positive)}")
    
    if len(extreme_negative) > 0:
        print("\nExtreme negative days:")
        for _, row in extreme_negative.head().iterrows():
            print(f"  {row['date']}: {row['daily_return']*100:.1f}% "
                  f"(${row['portfolio_value_before']:,.0f} -> ${row['portfolio_value_after']:,.0f})")
    
    if len(extreme_positive) > 0:
        print("\nExtreme positive days:")
        for _, row in extreme_positive.head().iterrows():
            print(f"  {row['date']}: {row['daily_return']*100:.1f}% "
                  f"(${row['portfolio_value_before']:,.0f} -> ${row['portfolio_value_after']:,.0f})")
    
    # Find where portfolio goes negative
    negative_portfolio = df[df['portfolio_value_after'] < 0]
    print(f"\nDays with negative portfolio value: {len(negative_portfolio)}")
    
    if len(negative_portfolio) > 0:
        print("First negative portfolio day:")
        first_negative = negative_portfolio.iloc[0]
        print(f"  {first_negative['date']}: ${first_negative['portfolio_value_after']:,.0f}")
    
    # Check for NaN or infinite values
    nan_returns = df[df['daily_return'].isna()]
    inf_returns = df[np.isinf(df['daily_return'])]
    
    print(f"\nData quality issues:")
    print(f"NaN returns: {len(nan_returns)}")
    print(f"Infinite returns: {len(inf_returns)}")
    
    # Calculate proper Sharpe ratio
    returns_series = df['daily_return']
    annual_return = returns_series.mean() * 252
    annual_volatility = returns_series.std() * np.sqrt(252)
    sharpe = (annual_return - 0.02) / annual_volatility if annual_volatility > 0 else 0.0
    
    print(f"Sharpe ratio calculation:")
    print(f"  Annual return: {annual_return:.4f} ({annual_return*100:.2f}%)")
    print(f"  Annual volatility: {annual_volatility:.4f} ({annual_volatility*100:.2f}%)")
    print(f"  Sharpe ratio: {sharpe:.4f}")

if __name__ == "__main__":
    analyze_portfolio_trajectory()