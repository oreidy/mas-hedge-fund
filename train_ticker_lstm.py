"""
Improved LSTM training pipeline for individual ticker return prediction.

This script eliminates look-ahead bias and trains models to predict individual ticker returns
using only historical consensus tickers and proper temporal validation.

Usage:
    python train_ticker_lstm.py --data-dir training_data/episodes --output-dir models/ticker_lstm
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from src.rl.data_collection.episode import TrainingEpisode
from src.rl.preprocessing.ticker_preprocessor import TickerPreprocessor
from src.rl.models.ticker_lstm_model import TickerLSTMModel
from src.rl.training.ticker_trainer import TickerLSTMTrainer
from utils.logger import setup_logger, logger


def load_episodes_from_directory(data_dir: str) -> List[TrainingEpisode]:
    """
    Load training episodes from JSON files in directory.
    
    Args:
        data_dir: Directory containing episode JSON files
        
    Returns:
        List of TrainingEpisode objects sorted by date
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    episode_files = list(data_path.glob("episode_*.json"))
    if not episode_files:
        raise FileNotFoundError(f"No episode files found in {data_dir}")
    
    logger.info(f"Loading {len(episode_files)} episode files from {data_dir}")
    
    episodes = []
    for file_path in episode_files:
        try:
            with open(file_path, 'r') as f:
                episode_data = json.load(f)
            
            # Convert to TrainingEpisode object
            episode = TrainingEpisode.from_dict(episode_data)
            episodes.append(episode)
            
        except Exception as e:
            logger.warning(f"Failed to load episode from {file_path}: {e}")
            continue
    
    if not episodes:
        raise ValueError("No valid episodes loaded")
    
    # Sort by date to ensure proper temporal ordering
    episodes.sort(key=lambda ep: ep.date)
    
    logger.info(f"Successfully loaded {len(episodes)} episodes from {episodes[0].date} to {episodes[-1].date}")
    return episodes


def create_experiment_config(args) -> Dict[str, Any]:
    """Create experiment configuration dictionary."""
    return {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'sequence_length': args.sequence_length,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'patience': args.patience,
        'validation_split': args.validation_split,
        'min_consensus_strength': args.min_consensus_strength,
        'min_episodes_per_ticker': args.min_episodes_per_ticker,
        'walk_forward': args.walk_forward,
        'initial_train_size': args.initial_train_size,
        'retrain_frequency': args.retrain_frequency,
        'timestamp': datetime.now().isoformat(),
        'device': 'auto'
    }


def main():
    parser = argparse.ArgumentParser(description="Train improved LSTM model for ticker return prediction")
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='training_data/episodes',
                       help='Directory containing episode JSON files')
    parser.add_argument('--output-dir', type=str, default='models/ticker_lstm',
                       help='Directory to save trained models and results')
    
    # Model architecture arguments
    parser.add_argument('--sequence-length', type=int, default=20,
                       help='Length of input sequences for LSTM')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='LSTM hidden state size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate for regularization')
    
    # Training arguments
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--max-epochs', type=int, default=100,
                       help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Fraction of data for temporal validation')
    
    # Preprocessing arguments
    parser.add_argument('--min-consensus-strength', type=float, default=0.5,
                       help='Minimum signal strength to consider ticker as consensus')
    parser.add_argument('--min-episodes-per-ticker', type=int, default=10,
                       help='Minimum episodes required for ticker to be included')
    
    # Walk-forward validation arguments
    parser.add_argument('--walk-forward', action='store_true',
                       help='Perform walk-forward validation instead of simple train/test')
    parser.add_argument('--initial-train-size', type=int, default=250,
                       help='Initial training set size for walk-forward validation')
    parser.add_argument('--retrain-frequency', type=int, default=21,
                       help='How often to retrain in walk-forward validation (episodes)')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (cuda/cpu/mps/auto)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save training plots')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(debug_mode=args.debug)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create experiment config
    config = create_experiment_config(args)
    
    # Save config
    with open(output_path / 'experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Starting improved LSTM training pipeline")
    logger.info(f"Configuration: {config}")
    
    try:
        # Load episodes
        episodes = load_episodes_from_directory(args.data_dir)
        
        # Create preprocessor
        logger.info("Creating and fitting preprocessor")
        preprocessor = TickerPreprocessor(
            sequence_length=args.sequence_length,
            prediction_horizon=1,  # Always predict next-day returns
            min_consensus_strength=args.min_consensus_strength,
            min_episodes_per_ticker=args.min_episodes_per_ticker
        )
        
        # Fit preprocessor and get data shape
        logger.info("Fitting preprocessor on training episodes")
        X, y, metadata = preprocessor.fit_transform(episodes)
        
        if len(X) == 0:
            raise ValueError("No training data generated - check episodes and preprocessor settings")
        
        logger.info(f"Generated {len(X)} training samples with shape: {X.shape}")
        logger.info(f"Found {len(preprocessor.historical_consensus_tickers)} historical consensus tickers")
        
        # Create model
        logger.info("Creating LSTM model")
        device = args.device if args.device != 'auto' else None
        
        model = TickerLSTMModel(
            input_size=X.shape[2],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=device
        )
        
        # Build ticker vocabulary
        all_tickers = set(meta['ticker'] for meta in metadata)
        model.build_ticker_vocabulary(all_tickers)
        
        # Create trainer
        trainer = TickerLSTMTrainer(
            model=model,
            preprocessor=preprocessor,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            validation_split=args.validation_split
        )
        
        # Training or walk-forward validation
        if args.walk_forward:
            logger.info("Starting walk-forward validation")
            results = trainer.walk_forward_validate(
                episodes=episodes,
                initial_train_size=args.initial_train_size,
                retrain_frequency=args.retrain_frequency
            )
            
            # Save walk-forward results
            results_path = output_path / 'walk_forward_results.json'
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Walk-forward results saved to {results_path}")
            
            # Also save as pickle for easier loading
            pickle_path = output_path / 'walk_forward_results.pkl'
            with open(pickle_path, 'wb') as f:
                pickle.dump(results, f)
                
        else:
            logger.info("Starting standard training with temporal validation")
            results = trainer.train(episodes)
            
            # Save training results
            results_path = output_path / 'training_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Training results saved to {results_path}")
        
        # Save model and preprocessor
        model_path = output_path / 'ticker_lstm_model.pth'
        model.save(str(model_path), include_optimizer=True, optimizer=trainer.optimizer)
        
        preprocessor_path = output_path / 'ticker_preprocessor.pkl'
        preprocessor.save(str(preprocessor_path))
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        # Save training plots if requested
        if args.save_plots and not args.walk_forward:
            plots_dir = output_path / 'plots'
            trainer.save_training_plots(str(plots_dir))
            logger.info(f"Training plots saved to {plots_dir}")
        
        # Print summary
        if args.walk_forward and 'final_metrics' in results:
            logger.info("Walk-forward validation completed successfully")
            logger.info(f"Final metrics: {results['final_metrics']}")
        elif not args.walk_forward:
            logger.info("Training completed successfully")
            logger.info(f"Best validation loss: {results['best_val_loss']:.6f}")
            logger.info(f"Final metrics: {results['final_metrics']}")
        
        logger.info(f"All outputs saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()