"""
Data preprocessing pipeline for LSTM training.

This module handles the conversion of raw training episodes into 
sequences suitable for LSTM model training.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from pathlib import Path

from ..data_collection.episode import TrainingEpisode
from .features import FeatureExtractor
from .scalers import create_scalers, apply_scalers
from utils.logger import logger


class DataPreprocessor:
    """
    Processes training episodes into sequences suitable for LSTM training.
    
    This class handles:
    1. Converting categorical signals to numeric values
    2. Extracting and engineering features
    3. Creating sequences for temporal modeling
    4. Scaling features appropriately
    5. Preparing data for PyTorch training
    """
    
    def __init__(
        self,
        sequence_length: int = 30,
        prediction_horizon: int = 1,
        feature_scaler: str = "standard",
        target_scaler: str = "minmax",
        include_market_features: bool = True,
        include_consensus_features: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            sequence_length: Length of input sequences for LSTM
            prediction_horizon: Number of steps ahead to predict
            feature_scaler: Type of scaling for features ('standard' or 'minmax')
            target_scaler: Type of scaling for targets ('standard' or 'minmax')
            include_market_features: Whether to include market data features
            include_consensus_features: Whether to include signal consensus features
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_scaler_type = feature_scaler
        self.target_scaler_type = target_scaler
        self.include_market_features = include_market_features
        self.include_consensus_features = include_consensus_features
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.scalers = {}
        self.feature_names = []
        self.is_fitted = False
        
        # Signal encoding
        self.signal_mapping = {
            'bullish': 1.0,
            'neutral': 0.0,
            'bearish': -1.0
        }
        
        # Action encoding
        self.action_mapping = {
            'buy': 1.0,
            'sell': -1.0,
            'short': -2.0,
            'cover': 2.0,
            'hold': 0.0
        }
        
        logger.info(f"DataPreprocessor initialized with sequence_length={sequence_length}, "
                   f"prediction_horizon={prediction_horizon}")
    
    def fit(self, episodes: List[TrainingEpisode]) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training episodes.
        
        Args:
            episodes: List of training episodes
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting preprocessor on {len(episodes)} episodes")
        
        # Extract features from all episodes
        features_df = self._extract_features_dataframe(episodes)
        
        # Create and fit scalers
        self.scalers = create_scalers(
            features_df,
            feature_scaler=self.feature_scaler_type,
            target_scaler=self.target_scaler_type
        )
        
        # Store feature names
        self.feature_names = [col for col in features_df.columns if col not in ['date', 'portfolio_return']]
        
        self.is_fitted = True
        logger.info(f"Preprocessor fitted successfully with {len(self.feature_names)} features")
        
        return self
    
    def transform(self, episodes: List[TrainingEpisode]) -> Dict[str, np.ndarray]:
        """
        Transform episodes into sequences for training.
        
        Args:
            episodes: List of training episodes
            
        Returns:
            Dictionary containing:
            - 'sequences': Input sequences (batch_size, sequence_length, n_features)
            - 'targets': Target values (batch_size, prediction_horizon)
            - 'masks': Sequence masks (batch_size, sequence_length)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transformation")
        
        logger.info(f"Transforming {len(episodes)} episodes into sequences")
        
        # Extract features
        features_df = self._extract_features_dataframe(episodes)
        
        # Apply scaling
        scaled_features = apply_scalers(features_df, self.scalers)
        
        # Create sequences
        sequences, targets, masks = self._create_sequences(scaled_features)
        
        logger.info(f"Created {len(sequences)} sequences of length {self.sequence_length}")
        
        return {
            'sequences': sequences,
            'targets': targets,
            'masks': masks,
            'feature_names': self.feature_names
        }
    
    def fit_transform(self, episodes: List[TrainingEpisode]) -> Dict[str, np.ndarray]:
        """
        Fit the preprocessor and transform episodes in one step.
        
        Args:
            episodes: List of training episodes
            
        Returns:
            Transformed data dictionary
        """
        return self.fit(episodes).transform(episodes)
    
    def _extract_features_dataframe(self, episodes: List[TrainingEpisode]) -> pd.DataFrame:
        """
        Extract features from episodes and create a DataFrame.
        
        Args:
            episodes: List of training episodes
            
        Returns:
            DataFrame with features for each episode
        """
        data_rows = []
        
        for episode in episodes:
            # Basic episode data
            row = {
                'date': episode.date,
                'portfolio_return': episode.portfolio_return,
                'cash_before': episode.portfolio_before.cash,
                'cash_after': episode.portfolio_after.cash,
                'cash_change': episode.portfolio_after.cash - episode.portfolio_before.cash,
            }
            
            # Market data features
            if self.include_market_features and episode.market_data:
                market_data = episode.market_data
                row.update({
                    'long_exposure': market_data.get('long_exposure', 0),
                    'short_exposure': market_data.get('short_exposure', 0),
                    'gross_exposure': market_data.get('gross_exposure', 0),
                    'net_exposure': market_data.get('net_exposure', 0),
                })
            
            # Aggregate signal features
            signal_features = self._aggregate_signal_features(episode)
            row.update(signal_features)
            
            # Decision features
            decision_features = self._aggregate_decision_features(episode)
            row.update(decision_features)
            
            # Consensus features
            if self.include_consensus_features:
                consensus_features = self._extract_consensus_features(episode)
                row.update(consensus_features)
            
            # Portfolio composition features
            portfolio_features = self._extract_portfolio_features(episode)
            row.update(portfolio_features)
            
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def _aggregate_signal_features(self, episode: TrainingEpisode) -> Dict[str, float]:
        """Extract aggregated signal features from an episode."""
        features = {}
        
        # Per-agent aggregations
        for agent_name, ticker_signals in episode.agent_signals.items():
            if not ticker_signals:
                continue
                
            # Convert signals to numeric
            numeric_signals = []
            confidences = []
            
            for ticker, signal in ticker_signals.items():
                numeric_signal = self.signal_mapping.get(signal.signal, 0.0)
                numeric_signals.append(numeric_signal)
                confidences.append(signal.confidence)
            
            if numeric_signals:
                features[f'{agent_name}_signal_mean'] = np.mean(numeric_signals)
                features[f'{agent_name}_signal_std'] = np.std(numeric_signals)
                features[f'{agent_name}_confidence_mean'] = np.mean(confidences)
                features[f'{agent_name}_confidence_std'] = np.std(confidences)
                features[f'{agent_name}_bullish_pct'] = np.mean([s > 0 for s in numeric_signals])
                features[f'{agent_name}_bearish_pct'] = np.mean([s < 0 for s in numeric_signals])
        
        # Overall signal aggregations
        all_signals = []
        all_confidences = []
        
        for agent_name, ticker_signals in episode.agent_signals.items():
            for ticker, signal in ticker_signals.items():
                numeric_signal = self.signal_mapping.get(signal.signal, 0.0)
                all_signals.append(numeric_signal)
                all_confidences.append(signal.confidence)
        
        if all_signals:
            features['overall_signal_mean'] = np.mean(all_signals)
            features['overall_signal_std'] = np.std(all_signals)
            features['overall_confidence_mean'] = np.mean(all_confidences)
            features['overall_confidence_std'] = np.std(all_confidences)
            features['overall_bullish_pct'] = np.mean([s > 0 for s in all_signals])
            features['overall_bearish_pct'] = np.mean([s < 0 for s in all_signals])
            features['signal_diversity'] = len(set(all_signals)) / len(all_signals) if all_signals else 0
        
        return features
    
    def _aggregate_decision_features(self, episode: TrainingEpisode) -> Dict[str, float]:
        """Extract aggregated decision features from an episode."""
        features = {}
        
        actions = []
        quantities = []
        confidences = []
        
        for ticker, decision in episode.llm_decisions.items():
            action_numeric = self.action_mapping.get(decision.action, 0.0)
            actions.append(action_numeric)
            quantities.append(decision.quantity)
            confidences.append(decision.confidence)
        
        if actions:
            features['decision_action_mean'] = np.mean(actions)
            features['decision_action_std'] = np.std(actions)
            features['decision_quantity_mean'] = np.mean(quantities)
            features['decision_quantity_std'] = np.std(quantities)
            features['decision_confidence_mean'] = np.mean(confidences)
            features['decision_confidence_std'] = np.std(confidences)
            features['buy_pct'] = np.mean([a > 0 for a in actions])
            features['sell_pct'] = np.mean([a < 0 for a in actions])
            features['hold_pct'] = np.mean([a == 0 for a in actions])
            features['total_quantity'] = np.sum(np.abs(quantities))
        
        return features
    
    def _extract_consensus_features(self, episode: TrainingEpisode) -> Dict[str, float]:
        """Extract consensus features for all tickers in the episode."""
        features = {}
        
        consensus_strengths = []
        consensus_agreements = []
        avg_confidences = []
        
        for ticker in episode.llm_decisions.keys():
            consensus = episode.get_signal_consensus(ticker)
            consensus_strengths.append(consensus['strength'])
            consensus_agreements.append(consensus['agreement'])
            avg_confidences.append(consensus['avg_confidence'])
        
        if consensus_strengths:
            features['consensus_strength_mean'] = np.mean(consensus_strengths)
            features['consensus_strength_std'] = np.std(consensus_strengths)
            features['consensus_agreement_mean'] = np.mean(consensus_agreements)
            features['consensus_agreement_std'] = np.std(consensus_agreements)
            features['consensus_confidence_mean'] = np.mean(avg_confidences)
            features['consensus_confidence_std'] = np.std(avg_confidences)
        
        return features
    
    def _extract_portfolio_features(self, episode: TrainingEpisode) -> Dict[str, float]:
        """Extract portfolio composition features."""
        features = {}
        
        # Portfolio before
        before_positions = episode.portfolio_before.positions
        long_positions = sum(pos['long'] for pos in before_positions.values())
        short_positions = sum(pos['short'] for pos in before_positions.values())
        
        features['positions_before_long'] = long_positions
        features['positions_before_short'] = short_positions
        features['positions_before_total'] = long_positions + short_positions
        
        # Portfolio after
        after_positions = episode.portfolio_after.positions
        long_positions_after = sum(pos['long'] for pos in after_positions.values())
        short_positions_after = sum(pos['short'] for pos in after_positions.values())
        
        features['positions_after_long'] = long_positions_after
        features['positions_after_short'] = short_positions_after
        features['positions_after_total'] = long_positions_after + short_positions_after
        
        # Position changes
        features['position_change_long'] = long_positions_after - long_positions
        features['position_change_short'] = short_positions_after - short_positions
        features['position_change_total'] = features['position_change_long'] + features['position_change_short']
        
        return features
    
    def _create_sequences(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences from the features DataFrame.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Tuple of (sequences, targets, masks)
        """
        # Sort by date
        features_df = features_df.sort_values('date').reset_index(drop=True)
        
        # Extract feature columns (exclude date and target)
        feature_cols = [col for col in features_df.columns if col not in ['date', 'portfolio_return']]
        X = features_df[feature_cols].values
        y = features_df['portfolio_return'].values
        
        sequences = []
        targets = []
        masks = []
        
        # Create sequences
        for i in range(len(X) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            seq = X[i:i + self.sequence_length]
            
            # Target (future returns)
            target = y[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
            
            # Mask (all ones for now, can be used for variable length sequences)
            mask = np.ones(self.sequence_length)
            
            sequences.append(seq)
            targets.append(target)
            masks.append(mask)
        
        return np.array(sequences), np.array(targets), np.array(masks)
    
    def save(self, filepath: Path) -> None:
        """Save the fitted preprocessor to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        save_data = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'feature_scaler_type': self.feature_scaler_type,
            'target_scaler_type': self.target_scaler_type,
            'include_market_features': self.include_market_features,
            'include_consensus_features': self.include_consensus_features,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'signal_mapping': self.signal_mapping,
            'action_mapping': self.action_mapping
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'DataPreprocessor':
        """Load a fitted preprocessor from disk."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create preprocessor with saved parameters
        preprocessor = cls(
            sequence_length=save_data['sequence_length'],
            prediction_horizon=save_data['prediction_horizon'],
            feature_scaler=save_data['feature_scaler_type'],
            target_scaler=save_data['target_scaler_type'],
            include_market_features=save_data['include_market_features'],
            include_consensus_features=save_data['include_consensus_features']
        )
        
        # Restore fitted state
        preprocessor.scalers = save_data['scalers']
        preprocessor.feature_names = save_data['feature_names']
        preprocessor.signal_mapping = save_data['signal_mapping']
        preprocessor.action_mapping = save_data['action_mapping']
        preprocessor.is_fitted = True
        
        logger.info(f"Preprocessor loaded from {filepath}")
        
        return preprocessor