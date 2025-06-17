#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime
from src.backtester import Backtester
from src.main import run_hedge_fund

def test_backtester():
    """Test the backtester with minimal configuration"""
    
    # Configuration
    tickers = ["WMT"]
    start_date = "2024-04-21"
    end_date = "2024-04-22"
    initial_capital = 100000
    selected_analysts = ["macro_analyst"]
    model_name = "llama3-70b-8192"
    model_provider = "Groq"
    
    print(f"Testing backtester from {start_date} to {end_date}")
    print(f"Tickers: {tickers}")
    print(f"Initial capital: ${initial_capital:,}")
    print(f"Analysts: {selected_analysts}")
    print()
    
    # Create backtester
    backtester = Backtester(
        agent=run_hedge_fund,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        model_name=model_name,
        model_provider=model_provider,
        selected_analysts=selected_analysts,
        initial_margin_requirement=0.0,
        debug_mode=True,
        verbose_data=True,
    )
    
    try:
        # Run the backtest
        print("Starting backtest...")
        performance_metrics = backtester.run_backtest()
        
        print("Backtest completed successfully!")
        print("Performance metrics:", performance_metrics)
        
        # Analyze performance
        performance_df = backtester.analyze_performance()
        
        return True, None
        
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    success, error = test_backtester()
    
    if success:
        print("✅ Backtester test completed successfully")
    else:
        print(f"❌ Backtester test failed: {error}")