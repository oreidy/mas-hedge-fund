"""
Model Manager for Walk-Forward LSTM Backtesting.

This module manages loading and caching of trained LSTM models for different
time periods in walk-forward backtesting.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..models.ticker_lstm_model import TickerLSTMModel
from ..preprocessing.ticker_preprocessor import TickerPreprocessor
from ..agents.lstm_decision_maker import LSTMDecisionMaker
from utils.logger import logger


@dataclass
class ModelInfo:
    """Information about a trained model."""
    model_name: str
    model_path: str
    preprocessor_path: str
    train_end_episode: int
    training_episodes_used: int
    training_end_date: str
    best_val_loss: float
    epochs_trained: int
    timestamp: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Create ModelInfo from dictionary."""
        return cls(
            model_name=data['model_name'],
            model_path=data['model_path'],
            preprocessor_path=data['preprocessor_path'],
            train_end_episode=data['train_end_episode'],
            training_episodes_used=data['training_episodes_used'],
            training_end_date=data['training_end_date'],
            best_val_loss=data['best_val_loss'],
            epochs_trained=data['epochs_trained'],
            timestamp=data['timestamp']
        )


class WalkForwardModelManager:
    """
    Manages trained LSTM models for walk-forward backtesting.
    
    This class handles:
    - Loading model metadata
    - Selecting appropriate models for given episodes
    - Caching loaded models for performance
    - Managing model transitions
    """
    
    def __init__(
        self,
        models_dir: str,
        cache_size: int = 3,
        device: str = None
    ):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory containing trained models and metadata
            cache_size: Maximum number of models to keep in memory
            device: Device to load models on (auto-detect if None)
        """
        self.models_dir = Path(models_dir)
        self.cache_size = cache_size
        self.device = device
        
        # Model cache: episode -> (model, preprocessor, decision_maker)
        self.model_cache = {}
        self.cache_access_order = []  # For LRU eviction
        
        # Model metadata
        self.model_metadata = {}
        self.sorted_episode_indices = []
        
        self._load_metadata()
        
        logger.info(f"WalkForwardModelManager initialized")
        logger.info(f"Models directory: {models_dir}")
        logger.info(f"Available models: {len(self.model_metadata)}")
        logger.info(f"Cache size: {cache_size}")
    
    def _load_metadata(self):
        """Load model metadata from the models directory."""
        metadata_path = self.models_dir / "model_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            raw_metadata = json.load(f)
        
        # Convert to ModelInfo objects
        self.model_metadata = {}
        for episode_str, model_data in raw_metadata.items():
            episode_idx = int(episode_str)
            self.model_metadata[episode_idx] = ModelInfo.from_dict(model_data)
        
        # Sort episode indices for efficient lookup
        self.sorted_episode_indices = sorted(self.model_metadata.keys())
        
        logger.info(f"Loaded metadata for {len(self.model_metadata)} models")
        logger.info(f"Episode range: {self.sorted_episode_indices[0]} to {self.sorted_episode_indices[-1]}")
    
    def get_model_for_episode(self, episode_index: int) -> Optional[ModelInfo]:
        """
        Get the appropriate model info for a given episode index.
        
        Args:
            episode_index: Episode index in the sequence
            
        Returns:
            ModelInfo for the latest model trained before this episode, or None
        """
        # Find the latest model that was trained before or at this episode
        available_models = [ep for ep in self.sorted_episode_indices if ep <= episode_index]
        
        if not available_models:
            return None
        
        # Get the most recent model
        latest_model_episode = max(available_models)
        return self.model_metadata[latest_model_episode]
    
    def load_model_components(
        self,
        model_info: ModelInfo
    ) -> Tuple[TickerLSTMModel, TickerPreprocessor, LSTMDecisionMaker]:
        """
        Load model, preprocessor, and decision maker from disk.
        
        Args:
            model_info: Model information
            
        Returns:
            Tuple of (model, preprocessor, decision_maker)
        """
        logger.info(f"Loading model components for {model_info.model_name}")
        
        # Load model
        model = TickerLSTMModel.load(model_info.model_path, device=self.device)
        
        # Load preprocessor
        preprocessor = TickerPreprocessor.load(model_info.preprocessor_path)
        
        # Create decision maker
        decision_maker = LSTMDecisionMaker(model=model, preprocessor=preprocessor)
        
        logger.info(f"Successfully loaded {model_info.model_name}")
        logger.info(f"Model has {len(decision_maker.valid_tickers)} valid tickers")
        
        return model, preprocessor, decision_maker
    
    def get_decision_maker_for_episode(self, episode_index: int) -> Optional[LSTMDecisionMaker]:
        """
        Get a decision maker for the given episode, with caching.
        
        Args:
            episode_index: Episode index in the sequence
            
        Returns:
            LSTMDecisionMaker instance, or None if no model available
        """
        # Check if already cached
        if episode_index in self.model_cache:
            # Update access order for LRU
            self.cache_access_order.remove(episode_index)
            self.cache_access_order.append(episode_index)
            
            _, _, decision_maker = self.model_cache[episode_index]
            logger.debug(f"Using cached model for episode {episode_index}")
            return decision_maker
        
        # Get model info
        model_info = self.get_model_for_episode(episode_index)
        if model_info is None:
            logger.warning(f"No model available for episode {episode_index}")
            return None
        
        # Load model components
        try:
            model, preprocessor, decision_maker = self.load_model_components(model_info)
            
            # Add to cache
            self._add_to_cache(episode_index, model, preprocessor, decision_maker)
            
            logger.info(f"Loaded model {model_info.model_name} for episode {episode_index}")
            return decision_maker
            
        except Exception as e:
            logger.error(f"Failed to load model for episode {episode_index}: {e}")
            return None
    
    def _add_to_cache(
        self,
        episode_index: int,
        model: TickerLSTMModel,
        preprocessor: TickerPreprocessor,
        decision_maker: LSTMDecisionMaker
    ):
        """Add model components to cache with LRU eviction."""
        # Remove oldest if cache is full
        while len(self.model_cache) >= self.cache_size:
            oldest_episode = self.cache_access_order.pop(0)
            del self.model_cache[oldest_episode]
            logger.debug(f"Evicted model for episode {oldest_episode} from cache")
        
        # Add new model
        self.model_cache[episode_index] = (model, preprocessor, decision_maker)
        self.cache_access_order.append(episode_index)
        
        logger.debug(f"Added model for episode {episode_index} to cache")
    
    def preload_models(self, episode_indices: list):
        """
        Preload models for given episode indices.
        
        Args:
            episode_indices: List of episode indices to preload
        """
        logger.info(f"Preloading models for {len(episode_indices)} episodes")
        
        for episode_idx in episode_indices:
            if episode_idx not in self.model_cache:
                decision_maker = self.get_decision_maker_for_episode(episode_idx)
                if decision_maker is None:
                    logger.warning(f"Could not preload model for episode {episode_idx}")
        
        logger.info(f"Preloading completed. Cache contains {len(self.model_cache)} models")
    
    def clear_cache(self):
        """Clear the model cache."""
        self.model_cache.clear()
        self.cache_access_order.clear()
        logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state."""
        return {
            'cached_episodes': list(self.model_cache.keys()),
            'cache_size': len(self.model_cache),
            'max_cache_size': self.cache_size,
            'access_order': self.cache_access_order.copy()
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of available models."""
        if not self.model_metadata:
            return {'no_models': True}
        
        episodes = list(self.model_metadata.keys())
        val_losses = [info.best_val_loss for info in self.model_metadata.values()]
        
        return {
            'total_models': len(self.model_metadata),
            'episode_range': (min(episodes), max(episodes)),
            'avg_val_loss': sum(val_losses) / len(val_losses),
            'best_val_loss': min(val_losses),
            'worst_val_loss': max(val_losses),
            'models_dir': str(self.models_dir),
            'cache_info': self.get_cache_info()
        }
    
    def validate_model_files(self) -> Dict[str, Any]:
        """
        Validate that all model files exist and are accessible.
        
        Returns:
            Validation results
        """
        results = {
            'total_models': len(self.model_metadata),
            'valid_models': 0,
            'missing_model_files': [],
            'missing_preprocessor_files': [],
            'validation_errors': []
        }
        
        for episode_idx, model_info in self.model_metadata.items():
            try:
                # Check model file
                if not os.path.exists(model_info.model_path):
                    results['missing_model_files'].append({
                        'episode': episode_idx,
                        'path': model_info.model_path
                    })
                    continue
                
                # Check preprocessor file
                if not os.path.exists(model_info.preprocessor_path):
                    results['missing_preprocessor_files'].append({
                        'episode': episode_idx,
                        'path': model_info.preprocessor_path
                    })
                    continue
                
                results['valid_models'] += 1
                
            except Exception as e:
                results['validation_errors'].append({
                    'episode': episode_idx,
                    'error': str(e)
                })
        
        logger.info(f"Model validation: {results['valid_models']}/{results['total_models']} models valid")
        
        return results