#!/usr/bin/env python3
"""Investigate extreme daily returns in LSTM backtest."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.append('src')

from rl.backtesting.lstm_backtest import LSTMWalkForwardBacktester

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format='%(levelname)s:%(name)s:%(message)s'
)

def investigate_extreme_returns():
    """Investigate what causes extreme daily returns."""
    
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
    
    # Load the CSV to find extreme days
    df = pd.read_csv('benchmarking/data/lstm_walkforward_returns_20250805_142149.csv')
    
    # Find worst days
    worst_days = df[df['return'] < -0.1].sort_values('return').head(5)
    print("5 worst return days:")
    for _, row in worst_days.iterrows():
        print(f"  {row['date']}: {row['return']*100:.2f}%")
    
    # Let's investigate one specific extreme day in detail
    extreme_date = "2024-11-21"  # -39.30% day
    print(f"\n=== Investigating {extreme_date} (-39.30%) ===")
    
    # Find the episode for this date
    target_episode = None
    episode_index = None
    for i, ep in enumerate(episodes):
        if ep.date == extreme_date:
            target_episode = ep
            episode_index = i
            break
    
    if target_episode is None:
        print(f"Could not find episode for {extreme_date}")
        return
    
    print(f"Found episode at index {episode_index}")
    
    # Run backtest up to this point with detailed logging
    print("\n=== Running backtest up to extreme day ===")
    
    # Enable more detailed logging for analysis
    import logging
    logging.getLogger('lstm_backtest').setLevel(logging.INFO)
    logging.getLogger('portfolio_engine').setLevel(logging.INFO)
    
    # Run backtest with limited episodes (up to and including the extreme day)
    limited_episodes = episodes[:episode_index + 1]
    
    # Let's manually step through the portfolio engine to see what happens
    from rl.backtesting.portfolio_engine import PortfolioEngine
    from rl.backtesting.model_manager import WalkForwardModelManager
    
    # Initialize portfolio engine like in the backtest
    all_consensus_tickers = set()
    for episode in limited_episodes:
        risk_mgmt_signals = episode.agent_signals.get('risk_management_agent', {})
        if risk_mgmt_signals:
            all_consensus_tickers.update(risk_mgmt_signals.keys())
    
    all_consensus_tickers = sorted(list(all_consensus_tickers))
    print(f"Total consensus tickers: {len(all_consensus_tickers)}")
    
    portfolio_engine = PortfolioEngine(
        initial_capital=100000.0,
        all_tickers=all_consensus_tickers,
        max_positions=30,
        transaction_cost=0.0005,
        margin_ratio=0.0,  # This is key - no margin requirement!
        bond_etfs=["SHY", "TLT"]
    )
    
    # Load model manager
    model_manager = WalkForwardModelManager(
        models_dir="src/rl/trained_models",
        cache_size=3
    )
    
    # Run through a few episodes leading up to the extreme day
    start_index = max(250, episode_index - 5)  # Last 5 days before extreme day
    
    print(f"\n=== Portfolio state evolution (episodes {start_index} to {episode_index}) ===")
    
    # Skip to near the extreme day for efficiency
    for i in range(start_index, episode_index + 1):
        episode = limited_episodes[i]
        
        # Get pricing data
        execution_prices = episode.market_data.get('execution_prices', {})
        evaluation_prices = episode.market_data.get('evaluation_prices', {})
        
        # Get previous episode's evaluation prices
        previous_evaluation_prices = {}
        if i > 0:
            previous_episode = limited_episodes[i-1]
            previous_evaluation_prices = previous_episode.market_data.get('evaluation_prices', {})
        else:
            previous_evaluation_prices = execution_prices
        
        # Get portfolio value before
        portfolio_value_before = portfolio_engine.get_portfolio_value(previous_evaluation_prices)
        
        # Get consensus data
        consensus_data = backtester.extract_consensus_data(episode)
        consensus_tickers = consensus_data['consensus_tickers']
        risk_limits = consensus_data['risk_limits']
        
        # Show portfolio state
        portfolio_state = portfolio_engine.get_portfolio_state_copy()
        cash = portfolio_state['cash']
        
        # Count positions
        active_positions = 0
        total_long_value = 0
        total_short_value = 0
        
        for ticker, position in portfolio_state['positions'].items():
            if position['long'] > 0 or position['short'] > 0:
                active_positions += 1
                if ticker in evaluation_prices:
                    if position['long'] > 0:
                        total_long_value += position['long'] * evaluation_prices[ticker]
                    if position['short'] > 0:
                        total_short_value += position['short'] * evaluation_prices[ticker]
        
        print(f"\nEpisode {i} ({episode.date}):")
        print(f"  Portfolio value before: ${portfolio_value_before:,.0f}")
        print(f"  Cash: ${cash:,.0f} ({cash/portfolio_value_before*100:.1f}%)")
        print(f"  Active positions: {active_positions}")
        print(f"  Long value: ${total_long_value:,.0f}")
        print(f"  Short value: ${total_short_value:,.0f}")
        print(f"  Consensus tickers: {len(consensus_tickers)}")
        
        # If this is the extreme day, show more detail
        if i == episode_index:
            print(f"\n=== EXTREME DAY DETAIL ({episode.date}) ===")
            
            # Show individual positions
            print("Positions detail:")
            for ticker, position in portfolio_state['positions'].items():
                if position['long'] > 0 or position['short'] > 0:
                    current_price = evaluation_prices.get(ticker, 0)
                    if position['long'] > 0:
                        value = position['long'] * current_price
                        print(f"  {ticker}: {position['long']} shares LONG @ ${current_price:.2f} = ${value:,.0f}")
                    if position['short'] > 0:
                        value = position['short'] * current_price
                        print(f"  {ticker}: {position['short']} shares SHORT @ ${current_price:.2f} = ${value:,.0f}")
            
            # Calculate what the return should be based on individual stock moves
            print(f"\nStock price analysis:")
            # Get previous day's prices
            prev_episode = limited_episodes[i-1] if i > 0 else episode
            prev_evaluation_prices = prev_episode.market_data.get('evaluation_prices', {})
            
            total_position_pnl = 0
            for ticker, position in portfolio_state['positions'].items():
                if (position['long'] > 0 or position['short'] > 0) and ticker in evaluation_prices and ticker in prev_evaluation_prices:
                    current_price = evaluation_prices[ticker]
                    prev_price = prev_evaluation_prices[ticker]
                    price_change = (current_price - prev_price) / prev_price if prev_price > 0 else 0
                    
                    if position['long'] > 0:
                        position_pnl = position['long'] * (current_price - prev_price)
                        total_position_pnl += position_pnl
                        print(f"  {ticker}: {price_change*100:.2f}% price change, LONG P&L: ${position_pnl:,.0f}")
                    
                    if position['short'] > 0:
                        position_pnl = position['short'] * (prev_price - current_price)  # Short benefits from price decline
                        total_position_pnl += position_pnl
                        print(f"  {ticker}: {price_change*100:.2f}% price change, SHORT P&L: ${position_pnl:,.0f}")
            
            expected_portfolio_value = portfolio_value_before + total_position_pnl
            expected_return = total_position_pnl / portfolio_value_before if portfolio_value_before > 0 else 0
            
            print(f"\nExpected portfolio change: ${total_position_pnl:,.0f}")
            print(f"Expected portfolio value: ${expected_portfolio_value:,.0f}")
            print(f"Expected return: {expected_return*100:.2f}%")
        
        # Actually run the backtest step (but don't make any decisions to avoid changing portfolio)
        portfolio_value_after = portfolio_engine.get_portfolio_value(evaluation_prices)
        daily_return = portfolio_engine.calculate_portfolio_return(portfolio_value_before, portfolio_value_after)
        
        print(f"  Portfolio value after: ${portfolio_value_after:,.0f}")
        print(f"  Daily return: {daily_return*100:.2f}%")
        
        if i == episode_index:
            print(f"\n=== ROOT CAUSE ANALYSIS ===")
            if abs(daily_return) > 0.1:  # If return is extreme
                print(f"EXTREME RETURN DETECTED: {daily_return*100:.2f}%")
                print("Possible causes:")
                print("1. Portfolio value went negative or near zero")
                print("2. Massive short positions with adverse price moves")
                print("3. Calculation error in portfolio valuation")
                print("4. Missing or incorrect price data")
                
                # Check for negative portfolio values
                if portfolio_value_before <= 0:
                    print(f"ERROR: Portfolio value before is non-positive: ${portfolio_value_before:,.0f}")
                if portfolio_value_after <= 0:
                    print(f"ERROR: Portfolio value after is non-positive: ${portfolio_value_after:,.0f}")
                
                # Check for extreme position sizes
                for ticker, position in portfolio_state['positions'].items():
                    if ticker in evaluation_prices:
                        current_price = evaluation_prices[ticker]
                        if position['short'] > 0:
                            short_value = position['short'] * current_price
                            if short_value > portfolio_value_before:
                                print(f"EXTREME SHORT: {ticker} short value ${short_value:,.0f} > portfolio ${portfolio_value_before:,.0f}")

if __name__ == "__main__":
    investigate_extreme_returns()