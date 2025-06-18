#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime
from src.backtester import Backtester
from src.main import run_hedge_fund
from src.utils.logger import setup_logger

def test_backtester():
    """Test the backtester with minimal configuration"""
    
    # Simulate CLI argument flags (matching src/backtester.py CLI behavior)
    args_debug = True
    args_verbose_data = True  
    args_quiet = False
    
    # Set up logger exactly like CLI (src/backtester.py:799-803)
    setup_logger(
        debug_mode=args_debug and not args_quiet,
        log_to_file=False,
        log_file=None
    )
    
    # Set verbose_data flag exactly like CLI (src/backtester.py:805-806)
    verbose_data = args_verbose_data and args_debug and not args_quiet
    
    print(f"Debug logging enabled: {args_debug and not args_quiet}")
    print(f"Verbose data enabled: {verbose_data}")
    
    # Test debug logging to verify it's working
    from src.utils.logger import logger
    logger.debug("🔍 TEST: Debug logging is working in test_backtester.py")
    logger.info("ℹ️ TEST: Info logging is working in test_backtester.py")
    
    # Configuration - matching CLI: --ticker WMT --start-date 2022-04-21 --end-date 2022-04-22 --debug --verbose-data
    tickers = ["WMT"]
    start_date = "2022-04-21"
    end_date = "2022-04-22"
    initial_capital = 100000
    selected_analysts = ["macro_analyst"]  # Use just one analyst like original test
    model_name = "llama3-70b-8192"
    model_provider = "Groq"
    
    print(f"Testing backtester from {start_date} to {end_date}")
    print(f"Tickers: {tickers}")
    print(f"Initial capital: ${initial_capital:,}")
    print(f"Analysts: {selected_analysts}")
    print()
    
    # Create backtester exactly like CLI (src/backtester.py:864-876)
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
        debug_mode=args_debug and not args_quiet,
        verbose_data=args_verbose_data and args_debug and not args_quiet,
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