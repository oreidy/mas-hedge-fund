#!/usr/bin/env python3
"""Debug specific extreme return day."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.append('src')

from rl.backtesting.lstm_backtest import LSTMWalkForwardBacktester

def debug_specific_day():
    """Debug a specific day with extreme returns."""
    
    # Set up detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s]: %(message)s'
    )
    
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
    
    # Find the episode corresponding to 2024-11-21 (CSV row 414)
    # CSV starts from episode 250, so CSV row 414 = episode 250 + 414 = 664
    target_episode_index = 250 + 414  # = 664
    target_date = "2024-11-21"
    
    print(f"Target episode index: {target_episode_index}")
    print(f"Target date: {target_date}")
    
    if target_episode_index >= len(episodes):
        print(f"Episode index {target_episode_index} out of range (max: {len(episodes)-1})")
        return
    
    target_episode = episodes[target_episode_index]
    print(f"Actual episode date: {target_episode.date}")
    
    if target_episode.date != target_date:
        print(f"WARNING: Date mismatch! Expected {target_date}, got {target_episode.date}")
        # Find the correct episode
        for i, ep in enumerate(episodes):
            if ep.date == target_date:
                target_episode_index = i
                target_episode = ep
                print(f"Found correct episode at index {i}")
                break
        else:
            print(f"Could not find episode for {target_date}")
            return
    
    # Now run a targeted backtest up to and including this episode
    print(f"\n=== Running backtest to episode {target_episode_index} ===")
    
    # Run backtest on a subset that includes this day
    subset_episodes = episodes[:target_episode_index + 1]
    
    # Temporarily modify the backtester to add detailed logging for this specific day
    original_method = backtester.run_walk_forward_backtest
    
    def debug_backtest(episodes_list):
        results = original_method(episodes_list)
        
        # Find the result for our target date
        daily_results = results['daily_results']
        target_result = None
        for result in daily_results:
            if result['date'] == target_date:
                target_result = result
                break
        
        if target_result:
            print(f"\n=== DETAILED ANALYSIS FOR {target_date} ===")
            print(f"Portfolio value before: ${target_result['portfolio_value_before']:,.2f}")
            print(f"Portfolio value after: ${target_result['portfolio_value_after']:,.2f}")
            print(f"Daily return: {target_result['daily_return']*100:.4f}%")
            print(f"LSTM decisions: {len(target_result['lstm_decisions'])}")
            print(f"Executed trades: {len(target_result['executed_trades'])}")
            
            # Show LSTM decisions
            if target_result['lstm_decisions']:
                print(f"\nLSTM Decisions:")
                for ticker, decision in target_result['lstm_decisions'].items():
                    action = decision['action']
                    quantity = decision['quantity']
                    confidence = decision['confidence']
                    if action != 'hold':
                        print(f"  {ticker}: {action} {quantity} shares (confidence: {confidence:.2f})")
            
            # Show executed trades
            if target_result['executed_trades']:
                print(f"\nExecuted Trades:")
                for ticker, qty in target_result['executed_trades'].items():
                    print(f"  {ticker}: {qty} shares")
            
            # Check for problematic calculations
            if abs(target_result['daily_return']) > 0.1:
                print(f"\n*** EXTREME RETURN DETECTED: {target_result['daily_return']*100:.2f}% ***")
                
                value_before = target_result['portfolio_value_before']
                value_after = target_result['portfolio_value_after']
                value_change = value_after - value_before
                
                print(f"Value change: ${value_change:,.2f}")
                print(f"Percentage change: {(value_change/value_before)*100:.2f}%")
                
                if value_before <= 0:
                    print(f"ERROR: Portfolio value before is non-positive!")
                if value_after <= 0:
                    print(f"ERROR: Portfolio value after is non-positive!")
                
                # If we have a massive negative return, let's see if it's due to short positions
                if target_result['daily_return'] < -0.1:
                    print(f"MASSIVE LOSS detected - likely due to:")
                    print(f"1. Large short positions with adverse price moves")
                    print(f"2. Portfolio going deeply negative")
                    print(f"3. Calculation error when portfolio value approaches zero")
        
        return results
    
    # Replace the method temporarily
    backtester.run_walk_forward_backtest = debug_backtest
    
    try:
        results = backtester.run_walk_forward_backtest(subset_episodes)
        print(f"\nBacktest completed with {len(results['daily_results'])} results")
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_specific_day()