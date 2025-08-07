"""
Walk-Forward Model Training Pipeline.

This script pre-trains LSTM models at regular intervals for use in walk-forward backtesting.
Each model is trained on an expanding window of data and saved with metadata for later use.

Usage:
    python train_walkforward_models.py --data-dir training_data/episodes --output-dir models/walkforward
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rl.data_collection.episode import TrainingEpisode
from src.rl.preprocessing.ticker_preprocessor import TickerPreprocessor
from src.rl.models.ticker_lstm_model import TickerLSTMModel
from src.rl.training.ticker_trainer import TickerLSTMTrainer
from utils.logger import setup_logger, logger


class WalkForwardModelTrainer:
    """
    Trains LSTM models at regular intervals for walk-forward backtesting.
    
    This creates a series of models trained on expanding windows of data,
    allowing for realistic walk-forward validation.
    """
    
    def __init__(
        self,
        initial_train_size: int = 250,
        retrain_frequency: int = 21,  # Every 21 episodes (monthly)
        sequence_length: int = 20,
        output_dir: str = "models/walkforward"
    ):
        """
        Initialize the walk-forward trainer.
        
        Args:
            initial_train_size: Number of episodes for initial training
            retrain_frequency: How often to retrain (in episodes)
            sequence_length: LSTM sequence length
            output_dir: Directory to save trained models
        """
        self.initial_train_size = initial_train_size
        self.retrain_frequency = retrain_frequency
        self.sequence_length = sequence_length
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track model metadata
        self.model_metadata = {}
        
        logger.info(f"WalkForwardModelTrainer initialized")
        logger.info(f"Initial training size: {initial_train_size} episodes")
        logger.info(f"Retrain frequency: {retrain_frequency} episodes")
        logger.info(f"Output directory: {output_dir}")
    
    def load_episodes_from_directory(self, data_dir: str) -> List[TrainingEpisode]:
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
        
        episode_files = sorted(list(data_path.glob("episode_*.json")))
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
    
    def train_single_model(
        self,
        episodes: List[TrainingEpisode],
        model_name: str,
        **training_kwargs
    ) -> Tuple[Dict[str, Any], str, str]:
        """
        Train a single ticker LSTM model on given episodes.
        
        Args:
            episodes: Training episodes
            model_name: Name for the model
            **training_kwargs: Additional training parameters
            
        Returns:
            Tuple of (training_results, model_path, preprocessor_path)
        """
        logger.info(f"Training model '{model_name}' on {len(episodes)} episodes...")
        
        # Create preprocessor
        preprocessor = TickerPreprocessor(
            sequence_length=self.sequence_length,
            prediction_horizon=1,
            min_consensus_strength=training_kwargs.get('min_consensus_strength', 0.5),
            min_episodes_per_ticker=training_kwargs.get('min_episodes_per_ticker', 10)
        )
        
        # Fit preprocessor and get training data
        X, y, metadata = preprocessor.fit_transform(episodes)
        
        if len(X) == 0:
            raise ValueError(f"No training data generated from {len(episodes)} episodes")
        
        # Create model
        model = TickerLSTMModel(
            input_size=X.shape[2],
            hidden_size=training_kwargs.get('hidden_size', 128),
            num_layers=training_kwargs.get('num_layers', 2),
            dropout=training_kwargs.get('dropout', 0.2)
        )
        
        # Build ticker vocabulary
        all_tickers = set(meta['ticker'] for meta in metadata)
        model.build_ticker_vocabulary(all_tickers)
        
        # Create trainer
        trainer = TickerLSTMTrainer(
            model=model,
            preprocessor=preprocessor,
            learning_rate=training_kwargs.get('learning_rate', 0.001),
            batch_size=training_kwargs.get('batch_size', 32),
            max_epochs=training_kwargs.get('max_epochs', 50),
            patience=training_kwargs.get('patience', 10),
            validation_split=training_kwargs.get('validation_split', 0.2)
        )
        
        # Train the model
        training_results = trainer.train(episodes)
        
        # Save model and preprocessor
        model_path = self.output_dir / f"{model_name}.pth"
        preprocessor_path = self.output_dir / f"{model_name}_preprocessor.pkl"
        
        model.save(str(model_path))\n        preprocessor.save(str(preprocessor_path))
        
        logger.info(f"Model '{model_name}' training completed")
        logger.info(f"Best validation loss: {training_results['best_val_loss']:.6f}")
        logger.info(f"Valid tickers: {len(all_tickers)}")
        
        return training_results, str(model_path), str(preprocessor_path)
    
    def run_walk_forward_training(
        self,
        episodes: List[TrainingEpisode],
        **training_kwargs
    ) -> Dict[str, Any]:
        """
        Run walk-forward model training.
        
        Args:
            episodes: All episodes in chronological order
            **training_kwargs: Additional training parameters
            
        Returns:
            Training summary and model metadata
        """
        if len(episodes) < self.initial_train_size + self.retrain_frequency:
            raise ValueError(f"Need at least {self.initial_train_size + self.retrain_frequency} episodes")
        
        logger.info(f"Starting walk-forward training on {len(episodes)} episodes")
        
        training_points = []
        current_pos = self.initial_train_size
        
        # Generate training points
        while current_pos <= len(episodes):
            training_points.append(current_pos)
            current_pos += self.retrain_frequency
        
        logger.info(f"Will train {len(training_points)} models at episodes: {training_points}")
        
        # Train models at each point
        for i, train_end_episode in enumerate(training_points):
            train_episodes = episodes[:train_end_episode]
            model_name = f"model_episode_{train_end_episode}"
            
            logger.info(f"Training model {i+1}/{len(training_points)}: {model_name}")
            
            try:
                training_results, model_path, preprocessor_path = self.train_single_model(
                    train_episodes, model_name, **training_kwargs
                )
                
                # Store model metadata
                self.model_metadata[train_end_episode] = {
                    'model_name': model_name,
                    'model_path': model_path,
                    'preprocessor_path': preprocessor_path,
                    'train_end_episode': train_end_episode,
                    'training_episodes_used': len(train_episodes),
                    'training_end_date': train_episodes[-1].date,
                    'best_val_loss': training_results['best_val_loss'],
                    'epochs_trained': training_results['epochs_trained'],
                    'model_info': training_results['model_info'],
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Model {model_name} completed successfully")
                
            except Exception as e:
                logger.error(f"Failed to train model {model_name}: {e}")
                continue
        
        # Save metadata
        metadata_path = self.output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        # Create summary
        summary = {
            'total_models_trained': len(self.model_metadata),
            'initial_train_size': self.initial_train_size,
            'retrain_frequency': self.retrain_frequency,
            'sequence_length': self.sequence_length,
            'total_episodes': len(episodes),
            'training_points': training_points,
            'output_directory': str(self.output_dir),
            'metadata_file': str(metadata_path),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Walk-forward training completed!")
        logger.info(f"Trained {len(self.model_metadata)} models")
        logger.info(f"Summary saved to: {summary_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return summary
    
    def get_model_for_episode(self, episode_index: int) -> Dict[str, str]:
        """
        Get the appropriate model paths for a given episode index.
        
        Args:
            episode_index: Episode index in the sequence
            
        Returns:
            Dictionary with model_path and preprocessor_path, or None if no model available
        """
        # Find the latest model that was trained before this episode
        available_models = [ep for ep in self.model_metadata.keys() if ep <= episode_index]
        
        if not available_models:
            return None
        
        # Get the most recent model
        latest_model_episode = max(available_models)
        model_info = self.model_metadata[latest_model_episode]
        
        return {
            'model_path': model_info['model_path'],
            'preprocessor_path': model_info['preprocessor_path'],
            'model_name': model_info['model_name'],
            'training_end_date': model_info['training_end_date']
        }


def main():
    parser = argparse.ArgumentParser(description="Train walk-forward LSTM models")
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='training_data/episodes',
                       help='Directory containing episode JSON files')
    parser.add_argument('--output-dir', type=str, default='models/walkforward',
                       help='Directory to save trained models')
    
    # Training schedule arguments
    parser.add_argument('--initial-train-size', type=int, default=250,
                       help='Number of episodes for initial training')
    parser.add_argument('--retrain-frequency', type=int, default=21,
                       help='How often to retrain (in episodes)')
    
    # Model architecture arguments
    parser.add_argument('--sequence-length', type=int, default=20,
                       help='LSTM sequence length')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='LSTM hidden state size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--max-epochs', type=int, default=50,
                       help='Maximum epochs per model')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Validation split fraction')
    
    # Preprocessing arguments
    parser.add_argument('--min-consensus-strength', type=float, default=0.5,
                       help='Minimum consensus strength for tickers')
    parser.add_argument('--min-episodes-per-ticker', type=int, default=10,
                       help='Minimum episodes per ticker')
    
    # Other arguments
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(debug_mode=args.debug)
    
    # Create trainer
    trainer = WalkForwardModelTrainer(
        initial_train_size=args.initial_train_size,
        retrain_frequency=args.retrain_frequency,
        sequence_length=args.sequence_length,
        output_dir=args.output_dir
    )
    
    try:
        # Load episodes
        episodes = trainer.load_episodes_from_directory(args.data_dir)
        
        if len(episodes) < args.initial_train_size + args.retrain_frequency:
            logger.error(f"Insufficient episodes: need at least {args.initial_train_size + args.retrain_frequency}, got {len(episodes)}")
            return
        
        # Prepare training kwargs
        training_kwargs = {
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs,
            'patience': args.patience,
            'validation_split': args.validation_split,
            'min_consensus_strength': args.min_consensus_strength,
            'min_episodes_per_ticker': args.min_episodes_per_ticker
        }
        
        # Run walk-forward training
        summary = trainer.run_walk_forward_training(episodes, **training_kwargs)
        
        # Print summary
        print(f"\\nWalk-Forward Training Summary:")
        print(f"  Total models trained: {summary['total_models_trained']}")
        print(f"  Total episodes processed: {summary['total_episodes']}")
        print(f"  Initial training size: {summary['initial_train_size']}")
        print(f"  Retrain frequency: {summary['retrain_frequency']} episodes")
        print(f"  Output directory: {summary['output_directory']}")
        print(f"  Training completed at: {summary['timestamp']}")
        
    except Exception as e:
        logger.error(f"Walk-forward training failed: {e}")
        raise


if __name__ == "__main__":
    main()