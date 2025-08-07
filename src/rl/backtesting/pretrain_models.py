#!/usr/bin/env python3
"""
Pre-training script for LSTM Walk-Forward Models.

This script pre-trains all LSTM models needed for walk-forward backtesting,
making the actual backtesting much faster by eliminating training time.

Usage:
    python pretrain_models.py [--episodes-dir EPISODES_DIR] [--models-dir MODELS_DIR] 
                              [--initial-training EPISODES] [--retraining-freq DAYS]
                              [--sequence-length LENGTH] [--force-retrain]
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from rl.data_collection.episode import TrainingEpisode, AgentSignal, LLMDecision, PortfolioState
from rl.preprocessing.ticker_preprocessor import TickerPreprocessor
from rl.models.ticker_lstm_model import TickerLSTMModel
from rl.training.ticker_trainer import TickerLSTMTrainer

logger = logging.getLogger(__name__)


class LSTMPreTrainer:
    """
    Pre-trainer for LSTM walk-forward models.
    
    This class handles the complete pre-training pipeline:
    1. Load all training episodes
    2. Determine retraining points based on frequency
    3. Train models with expanding windows for each point
    4. Save models with proper metadata for WalkForwardModelManager
    """
    
    def __init__(
        self,
        episodes_dir: str = "training_data/episodes",
        models_dir: str = "src/rl/trained_models",
        initial_training_episodes: int = 250,
        retraining_frequency: int = 30,
        sequence_length: int = 20,
        force_retrain: bool = False
    ):
        """
        Initialize the pre-trainer.
        
        Args:
            episodes_dir: Directory containing training episodes
            models_dir: Directory to save trained models
            initial_training_episodes: Number of episodes for initial training
            retraining_frequency: How often to retrain (in episodes/days)
            sequence_length: LSTM sequence length
            force_retrain: If True, overwrite existing models
        """
        self.episodes_dir = Path(episodes_dir)
        self.models_dir = Path(models_dir)
        self.initial_training_episodes = initial_training_episodes
        self.retraining_frequency = retraining_frequency
        self.sequence_length = sequence_length
        self.force_retrain = force_retrain
        
        # Create models directory
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preprocessor (will be fitted during training)
        self.preprocessor = TickerPreprocessor(
            sequence_length=sequence_length,
            prediction_horizon=1,
            min_consensus_strength=0.5,
            min_episodes_per_ticker=10
        )
        
        # Track training results
        self.training_results = {}
        self.model_metadata = {}
        
        logger.info(f"LSTMPreTrainer initialized")
        logger.info(f"Episodes directory: {episodes_dir}")
        logger.info(f"Models directory: {models_dir}")
        logger.info(f"Initial training: {initial_training_episodes} episodes")
        logger.info(f"Retraining frequency: {retraining_frequency} episodes")
    
    def load_all_episodes(self) -> List[TrainingEpisode]:
        """Load all training episodes in chronological order."""
        json_files = sorted(list(self.episodes_dir.glob("*.json")))
        
        episodes = []
        failed_loads = 0
        
        logger.info(f"Loading {len(json_files)} episodes from {self.episodes_dir}")
        
        for json_file in tqdm(json_files, desc="Loading episodes"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Convert agent signals to proper format
                processed_agent_signals = {}
                for agent_name, signals in data['agent_signals'].items():
                    if agent_name == "risk_management_agent":
                        # Risk management data has different structure - keep as-is
                        processed_agent_signals[agent_name] = signals
                    else:
                        processed_signals = {}
                        for ticker, signal_data in signals.items():
                            if isinstance(signal_data, dict) and 'signal' in signal_data and 'confidence' in signal_data:
                                processed_signals[ticker] = AgentSignal.from_dict(signal_data)
                        
                        if processed_signals:
                            processed_agent_signals[agent_name] = processed_signals
                
                episode = TrainingEpisode(
                    date=data['date'],
                    agent_signals=processed_agent_signals,
                    llm_decisions={
                        ticker: LLMDecision.from_dict(decision_data)
                        for ticker, decision_data in data['llm_decisions'].items()
                    },
                    portfolio_before=PortfolioState.from_dict(data['portfolio_before']),
                    portfolio_after=PortfolioState.from_dict(data['portfolio_after']),
                    portfolio_return=data['portfolio_return'],
                    market_data=data.get('market_data', {}),
                    metadata=data.get('metadata', {})
                )
                episodes.append(episode)
                
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                failed_loads += 1
                continue
        
        logger.info(f"Successfully loaded {len(episodes)} episodes ({failed_loads} failed)")
        return episodes
    
    def determine_retraining_points(self, episodes: List[TrainingEpisode]) -> List[int]:
        """
        Determine which episode indices need model retraining.
        
        Args:
            episodes: All episodes in chronological order
            
        Returns:
            List of episode indices where models should be trained
        """
        total_episodes = len(episodes)
        
        if total_episodes < self.initial_training_episodes + self.sequence_length:
            raise ValueError(f"Need at least {self.initial_training_episodes + self.sequence_length} episodes")
        
        retraining_points = []
        
        # Initial training point
        retraining_points.append(self.initial_training_episodes)
        
        # Subsequent retraining points
        current_episode = self.initial_training_episodes
        while current_episode + self.retraining_frequency < total_episodes:
            current_episode += self.retraining_frequency
            retraining_points.append(current_episode)
        
        logger.info(f"Determined {len(retraining_points)} retraining points: {retraining_points}")
        return retraining_points
    
    def train_model_for_episode(
        self, 
        training_episodes: List[TrainingEpisode], 
        episode_index: int,
        model_name: str
    ) -> Tuple[TickerLSTMModel, Dict[str, Any]]:
        """
        Train LSTM model on given episodes.
        
        Args:
            training_episodes: Episodes to train on (expanding window)
            episode_index: Episode index this model will be used from
            model_name: Name for saving the model
            
        Returns:
            Trained model and training results
        """
        logger.info(f"Training model '{model_name}' on {len(training_episodes)} episodes (up to episode {episode_index})")
        
        # Fit preprocessor and get input size
        X, y, metadata = self.preprocessor.fit_transform(training_episodes)
        
        if len(X) == 0:
            raise ValueError(f"No training data generated from {len(training_episodes)} episodes")
        
        input_size = X.shape[2]
        
        # Initialize ticker LSTM model
        model = TickerLSTMModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        
        # Build ticker vocabulary
        all_tickers = set(meta['ticker'] for meta in metadata)
        model.build_ticker_vocabulary(all_tickers)
        
        # Initialize trainer
        trainer = TickerLSTMTrainer(
            model=model,
            preprocessor=self.preprocessor,
            learning_rate=1e-3,
            batch_size=32,
            max_epochs=50,
            patience=15,
            validation_split=0.2
        )
        
        # Train the model
        results = trainer.train(training_episodes)
        
        # Save the model and preprocessor
        model_path = self.models_dir / f"{model_name}.pth"
        preprocessor_path = self.models_dir / f"{model_name}_preprocessor.pkl"
        
        model.save(str(model_path))
        self.preprocessor.save(str(preprocessor_path))
        
        logger.info(f"Model training completed. Best validation loss: {results['best_val_loss']:.6f}")
        logger.info(f"Model saved to {model_path}, preprocessor saved to {preprocessor_path}")
        
        # Store metadata for this model
        end_date = training_episodes[-1].date if training_episodes else "unknown"
        
        self.model_metadata[episode_index] = {
            'model_name': model_name,
            'model_path': str(model_path),
            'preprocessor_path': str(preprocessor_path),
            'train_end_episode': episode_index,
            'training_episodes_used': len(training_episodes),
            'training_end_date': end_date,
            'best_val_loss': results['best_val_loss'],
            'epochs_trained': results.get('epochs_trained', 0),
            'timestamp': datetime.now().isoformat(),
            'total_tickers_trained': len(all_tickers),
            'valid_tickers': len(self.preprocessor.historical_consensus_tickers) if hasattr(self.preprocessor, 'historical_consensus_tickers') else 0
        }
        
        return model, results
    
    def save_metadata(self):
        """Save model metadata for WalkForwardModelManager."""
        metadata_path = self.models_dir / "model_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to {metadata_path}")
        logger.info(f"Metadata contains {len(self.model_metadata)} models")
    
    def check_existing_models(self, retraining_points: List[int]) -> List[int]:
        """
        Check which models already exist and don't need retraining.
        
        Args:
            retraining_points: All retraining points
            
        Returns:
            List of retraining points that need training
        """
        if self.force_retrain:
            logger.info("Force retrain enabled - will train all models")
            return retraining_points
        
        metadata_path = self.models_dir / "model_metadata.json"
        
        if not metadata_path.exists():
            logger.info("No existing metadata found - will train all models")
            return retraining_points
        
        try:
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
            
            existing_points = [int(ep) for ep in existing_metadata.keys()]
            missing_points = [ep for ep in retraining_points if ep not in existing_points]
            
            logger.info(f"Found {len(existing_points)} existing models")
            logger.info(f"Need to train {len(missing_points)} new models: {missing_points}")
            
            # Load existing metadata
            for ep_str, metadata in existing_metadata.items():
                self.model_metadata[int(ep_str)] = metadata
            
            return missing_points
            
        except Exception as e:
            logger.warning(f"Error reading existing metadata: {e}")
            logger.info("Will train all models")
            return retraining_points
    
    def run_pretraining(self, episodes: List[TrainingEpisode]) -> Dict[str, Any]:
        """
        Run the complete pre-training pipeline.
        
        Args:
            episodes: All episodes in chronological order
            
        Returns:
            Pre-training results and summary
        """
        logger.info(f"Starting LSTM pre-training on {len(episodes)} episodes")
        
        # Determine retraining points
        retraining_points = self.determine_retraining_points(episodes)
        
        # Check which models need training
        points_to_train = self.check_existing_models(retraining_points)
        
        if not points_to_train:
            logger.info("All models already exist and force_retrain=False. Nothing to train.")
            self.save_metadata()  # Ensure metadata is saved
            return {
                'total_models': len(retraining_points),
                'trained_models': 0,
                'skipped_models': len(retraining_points),
                'training_results': {}
            }
        
        # Train models for each retraining point
        trained_count = 0
        training_results = {}
        
        for episode_index in tqdm(points_to_train, desc="Training models"):
            # Get training data (expanding window up to this episode)
            training_episodes = episodes[:episode_index]
            model_name = f"model_episode_{episode_index}"
            
            try:
                model, results = self.train_model_for_episode(
                    training_episodes, episode_index, model_name
                )
                
                training_results[episode_index] = results
                trained_count += 1
                
                logger.info(f"Successfully trained model {trained_count}/{len(points_to_train)} for episode {episode_index}")
                
            except Exception as e:
                logger.error(f"Failed to train model for episode {episode_index}: {e}")
                continue
        
        # Save all metadata
        self.save_metadata()
        
        # Calculate summary statistics
        if training_results:
            val_losses = [r['best_val_loss'] for r in training_results.values()]
            avg_val_loss = sum(val_losses) / len(val_losses)
            best_val_loss = min(val_losses)
            worst_val_loss = max(val_losses)
        else:
            avg_val_loss = best_val_loss = worst_val_loss = 0.0
        
        summary = {
            'total_models': len(retraining_points),
            'trained_models': trained_count,
            'skipped_models': len(retraining_points) - len(points_to_train),
            'avg_val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'worst_val_loss': worst_val_loss,
            'training_results': training_results,
            'models_dir': str(self.models_dir),
            'episodes_processed': len(episodes)
        }
        
        logger.info("Pre-training completed!")
        logger.info(f"Total models: {summary['total_models']}")
        logger.info(f"Trained: {summary['trained_models']}, Skipped: {summary['skipped_models']}")
        if trained_count > 0:
            logger.info(f"Average validation loss: {avg_val_loss:.6f}")
            logger.info(f"Best validation loss: {best_val_loss:.6f}")
        
        return summary


def main():
    """Main function for pre-training LSTM models."""
    parser = argparse.ArgumentParser(description="Pre-train LSTM models for walk-forward backtesting")
    
    parser.add_argument(
        "--episodes-dir",
        type=str,
        default="training_data/episodes",
        help="Directory containing training episodes (default: training_data/episodes)"
    )
    
    parser.add_argument(
        "--models-dir",
        type=str,
        default="src/rl/trained_models",
        help="Directory to save trained models (default: src/rl/trained_models)"
    )
    
    parser.add_argument(
        "--initial-training",
        type=int,
        default=250,
        help="Number of episodes for initial training (default: 250)"
    )
    
    parser.add_argument(
        "--retraining-freq",
        type=int,
        default=30,
        help="Retraining frequency in episodes/days (default: 30)"
    )
    
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=20,
        help="LSTM sequence length (default: 20)"
    )
    
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining of all models, even if they exist"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize pre-trainer
    pretrainer = LSTMPreTrainer(
        episodes_dir=args.episodes_dir,
        models_dir=args.models_dir,
        initial_training_episodes=args.initial_training,
        retraining_frequency=args.retraining_freq,
        sequence_length=args.sequence_length,
        force_retrain=args.force_retrain
    )
    
    # Load episodes
    logger.info("Loading training episodes...")
    episodes = pretrainer.load_all_episodes()
    
    if len(episodes) < args.initial_training + args.sequence_length:
        logger.error(f"Insufficient episodes: {len(episodes)} (need at least {args.initial_training + args.sequence_length})")
        return 1
    
    # Run pre-training
    results = pretrainer.run_pretraining(episodes)
    
    # Print final summary
    print("\n" + "="*60)
    print("LSTM PRE-TRAINING SUMMARY")
    print("="*60)
    print(f"Episodes processed: {results['episodes_processed']}")
    print(f"Total models needed: {results['total_models']}")
    print(f"Models trained: {results['trained_models']}")
    print(f"Models skipped: {results['skipped_models']}")
    if results['trained_models'] > 0:
        print(f"Average validation loss: {results['avg_val_loss']:.6f}")
        print(f"Best validation loss: {results['best_val_loss']:.6f}")
        print(f"Worst validation loss: {results['worst_val_loss']:.6f}")
    print(f"Models saved to: {results['models_dir']}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())