#!/usr/bin/env python3

import sys
import os
import time

# Add src to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables from .env file FIRST
from dotenv import load_dotenv
load_dotenv()

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
    
    # Configuration for comprehensive agent testing - using tickers that showed agreeing signals
    tickers = ['ABBV', 'ALL', 'AZO', 'BMY', 'CINF', 'DGX', 'FAST', 'HOLX', 'INCY', 'INTC']  # 10 tickers from agreeing list
    start_date = "2022-04-05"   
    end_date = "2022-04-06"     # 2-day test to keep it fast but thorough
    initial_capital = 100000
    # Test ALL agents to ensure they're all working
    selected_analysts = [
        "technical_analyst", 
        "sentiment_analyst", 
        "fundamentals_analyst", 
        "valuation_analyst", 
        "warren_buffett", 
        "bill_ackman", 
        "macro_analyst", 
        "forward_looking_analyst"
    ]
    model_name = "llama-3.1-8b-instant"  # Use working Groq model
    model_provider = "Groq"
    screen_mode = False  # Disable screening mode to use specific tickers
    
    print(f"Testing ALL agents from {start_date} to {end_date}")
    print(f"Using {len(tickers)} tickers from agreeing list: {tickers}")
    print(f"Using screening mode: {screen_mode}")
    print(f"Initial capital: ${initial_capital:,}")
    print(f"Model: {model_name} ({model_provider})")
    print(f"Testing {len(selected_analysts)} analysts: {selected_analysts}")
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
        verbose_data=True,  # Enable verbose to see detailed logs
        screen_mode=screen_mode,
        collect_training_data=True,  # Enable training data collection
        training_data_dir="test_training_data",  # Directory for test data
        training_data_format="json"  # JSON format for easy inspection
    )
    
    try:
        # Test just the prefetch phase first to check our fix
        print("Testing prefetch phase with data validation...")
        start_time = time.time()
        
        # Run prefetch only to test our fix
        backtester.prefetch_data()
        
        prefetch_time = time.time() - start_time
        print(f"✅ Prefetch completed successfully in {prefetch_time:.2f} seconds")
        print("If no errors above, the fix is working!")
        
        # Run the full backtest to test fundamentals agent
        print("\nRunning full backtest to test fundamentals agent...")
        performance_metrics = backtester.run_backtest()
        
        total_time = time.time() - start_time
        
        print("=" * 50)
        print(f"🕐 Total backtest time: {total_time:.2f} seconds")
        print(f"🕐 Average time per day: {total_time / 2:.2f} seconds")  # 2 days
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