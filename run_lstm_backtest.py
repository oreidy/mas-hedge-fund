#!/usr/bin/env python3
"""
LSTM Backtesting Runner.

This script demonstrates how to use the complete LSTM backtesting system with
ticker-based predictions and walk-forward validation.

Usage:
    # Train walk-forward models first
    python train_walkforward_models.py --data-dir training_data/episodes --output-dir models/walkforward
    
    # Run LSTM backtest
    python run_lstm_backtest.py --models-dir models/walkforward --data-dir training_data/episodes
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rl.backtesting.lstm_backtest import LSTMWalkForwardBacktester
from utils.logger import setup_logger, logger


def main():
    parser = argparse.ArgumentParser(description="Run LSTM walk-forward backtesting")
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='training_data/episodes',
                       help='Directory containing episode JSON files')
    parser.add_argument('--models-dir', type=str, default='models/walkforward',
                       help='Directory containing pre-trained models (optional)')
    parser.add_argument('--output-dir', type=str, default='results/lstm_backtest',
                       help='Directory to save results')
    
    # Backtesting parameters
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                       help='Initial portfolio capital')
    parser.add_argument('--initial-train-size', type=int, default=250,
                       help='Initial training episodes')
    parser.add_argument('--retrain-frequency', type=int, default=21,
                       help='Retrain frequency (episodes)')
    parser.add_argument('--max-positions', type=int, default=30,
                       help='Maximum number of positions')
    parser.add_argument('--transaction-cost', type=float, default=0.0005,
                       help='Transaction cost (0.0005 = 5 basis points)')
    
    # Model parameters
    parser.add_argument('--sequence-length', type=int, default=20,
                       help='LSTM sequence length')
    
    # Other arguments
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export results to CSV for benchmarking')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(debug_mode=args.debug)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting LSTM walk-forward backtesting")
    logger.info(f"Configuration:")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Models directory: {args.models_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Initial capital: ${args.initial_capital:,.2f}")
    logger.info(f"  Initial training size: {args.initial_train_size} episodes")
    logger.info(f"  Retrain frequency: {args.retrain_frequency} episodes")
    logger.info(f"  Max positions: {args.max_positions}")
    
    try:
        # Create backtester
        backtester = LSTMWalkForwardBacktester(
            initial_capital=args.initial_capital,
            initial_training_episodes=args.initial_train_size,
            retraining_frequency=args.retrain_frequency,
            max_positions=args.max_positions,
            transaction_cost=args.transaction_cost,
            sequence_length=args.sequence_length,
            model_save_dir=args.models_dir
        )
        
        # Load episodes
        logger.info("Loading episodes...")
        episodes = backtester.load_all_episodes(args.data_dir)
        
        if len(episodes) < args.initial_train_size + args.retrain_frequency:
            logger.error(f"Insufficient episodes: need at least {args.initial_train_size + args.retrain_frequency}, got {len(episodes)}")
            return
        
        logger.info(f"Loaded {len(episodes)} episodes from {episodes[0].date} to {episodes[-1].date}")
        
        # Run walk-forward backtest
        logger.info("Starting walk-forward backtest...")
        results = backtester.run_walk_forward_backtest(episodes)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_path / f"lstm_backtest_results_{timestamp}.json"
        import json
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if hasattr(value, 'tolist'):
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Export to CSV for benchmarking if requested
        if args.export_csv:
            csv_file = backtester.export_results_to_csv(f"lstm_walkforward_returns_{timestamp}.csv")
            logger.info(f"Results exported to CSV: {csv_file}")
        
        # Print summary
        metrics = results['performance_metrics']
        print(f"\\n{'='*60}")
        print(f"LSTM Walk-Forward Backtest Results")
        print(f"{'='*60}")
        print(f"Episodes processed: {metrics['total_episodes']}")
        print(f"Initial capital: ${metrics['initial_capital']:,.2f}")
        print(f"Final portfolio value: ${metrics['final_portfolio_value']:,.2f}")
        print(f"Total return: {metrics['total_return']:.2%}")
        print(f"LSTM Sharpe ratio: {metrics['lstm_sharpe_ratio']:.3f}")
        print(f"LSTM volatility: {metrics['lstm_volatility']:.2%}")
        print(f"Models trained: {metrics['models_trained']}")
        print(f"Model retraining dates: {len(metrics['model_retraining_dates'])}")
        print(f"")
        print(f"Comparison to LLM baseline:")
        print(f"LLM total return: {metrics['llm_total_return']:.2%}")
        print(f"LLM Sharpe ratio: {metrics['llm_sharpe_ratio']:.3f}")
        print(f"LLM volatility: {metrics['llm_volatility']:.2%}")
        print(f"")
        print(f"Performance improvement:")
        if metrics['llm_total_return'] != 0:
            return_improvement = (metrics['total_return'] - metrics['llm_total_return']) / abs(metrics['llm_total_return'])
            print(f"Return improvement: {return_improvement:.1%}")
        
        if metrics['llm_sharpe_ratio'] != 0:
            sharpe_improvement = (metrics['lstm_sharpe_ratio'] - metrics['llm_sharpe_ratio']) / abs(metrics['llm_sharpe_ratio'])
            print(f"Sharpe ratio improvement: {sharpe_improvement:.1%}")
        
        print(f"{'='*60}")
        print(f"Results saved to: {output_path}")
        
        logger.info("LSTM backtesting completed successfully")
        
    except Exception as e:
        logger.error(f"LSTM backtesting failed: {e}")
        raise


if __name__ == "__main__":
    main()