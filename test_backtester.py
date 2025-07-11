#!/usr/bin/env python3

import sys
import os
import time

# Add src to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the logger module FIRST
from utils.logger import logger, LogLevel, setup_logger

# Configure the global logger instance BEFORE any other imports
logger.level = LogLevel.DEBUG
logger.debug_mode = True  # Enable debug mode for VTRS analysis

# Alternative: use setup_logger if it properly configures the global instance
setup_logger(debug_mode=True, log_to_file=False, log_file=None)

# Verify the logger is configured
print(f"Logger level after setup: {logger.level}")
print(f"Logger level value: {logger.level.value}")
print(f"DEBUG level value: {LogLevel.DEBUG.value}")

# NOW import other modules - they will use the already-configured logger
from datetime import datetime
from backtester import Backtester
from main import run_hedge_fund

def test_backtester():
    """Test the backtester with minimal configuration"""
    
    # Force logger reconfiguration to enable debug logging for VTRS analysis
    logger.set_level('DEBUG')
    
    # Add these debug messages to verify logger is working
    logger.debug("🔍 TEST: Debug logging is working in test_backtester.py", module="test_backtester")
    logger.info("ℹ️ TEST: Info logging is working in test_backtester.py", module="test_backtester")
    
    # Verify the logger level
    print(f"Logger level: {logger.level}")
    print(f"DEBUG level value: {logger.level.value}")
    from src.utils.logger import LogLevel
    print(f"LogLevel.DEBUG value: {LogLevel.DEBUG.value}")
    
    # Configuration for VTRS valuation agent testing
    tickers = ["VTRS"]  # VTRS ticker to reproduce astronomical confidence issue
    start_date = "2025-07-10"   # Day before yesterday  
    end_date = "2025-07-11"     # Yesterday
    initial_capital = 100000
    selected_analysts = ["valuation_analyst"]  # Only valuation agent to test extreme confidence issue
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
        # Run the backtest with timing analysis
        print("Starting backtest...")
        start_time = time.time()
        
        # Track individual component times
        print("📊 Timing Analysis:")
        print("=" * 50)
        
        performance_metrics = backtester.run_backtest()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("=" * 50)
        print(f"🕐 Total backtest time: {total_time:.2f} seconds")
        print(f"🕐 Average time per day: {total_time / 3:.2f} seconds")
        print("=" * 50)
        
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