"""
Improved data preprocessing pipeline for LSTM training with individual ticker predictions.

This module eliminates look-ahead bias by:
1. Using only pre-trade portfolio states for feature extraction
2. Training on individual ticker returns instead of portfolio returns
3. Using proper temporal ordering (train on episode T, predict episode T+1)
4. Only including tickers that were consensus in historical episodes
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Set
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import json
from pathlib import Path
from collections import defaultdict

from ..data_collection.episode import TrainingEpisode
from .features import FeatureExtractor
from .scalers import create_scalers, apply_scalers
from utils.logger import logger


class TickerPreprocessor:
    """
    Processes training episodes into ticker-specific sequences for LSTM training.
    
    Key improvements over original preprocessor:
    1. No look-ahead bias - uses only pre-trade features
    2. Individual ticker return prediction
    3. Proper temporal ordering
    4. Historical consensus ticker filtering
    """
    
    def __init__(
        self,
        sequence_length: int = 20,
        prediction_horizon: int = 1,
        feature_scaler: str = "standard",
        target_scaler: str = "minmax",
        min_consensus_strength: float = 0.5,
        min_episodes_per_ticker: int = 10
    ):
        """
        Initialize the ticker-based preprocessor.
        
        Args:
            sequence_length: Length of input sequences for LSTM
            prediction_horizon: Number of steps ahead to predict (always 1 for next-day returns)
            feature_scaler: Type of scaling for features ('standard' or 'minmax')
            target_scaler: Type of scaling for targets ('standard' or 'minmax')
            min_consensus_strength: Minimum signal strength to consider ticker as consensus
            min_episodes_per_ticker: Minimum episodes required for ticker to be included
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_scaler_type = feature_scaler
        self.target_scaler_type = target_scaler
        self.min_consensus_strength = min_consensus_strength
        self.min_episodes_per_ticker = min_episodes_per_ticker
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.scalers = {}
        self.feature_names = []
        self.is_fitted = False
        
        # Historical consensus tickers (learned from training data)
        self.historical_consensus_tickers = set()
        
        # Episode data cache for accessing raw JSON
        self._episode_data_cache = {}
        
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
        
        logger.info(f"TickerPreprocessor initialized with sequence_length={sequence_length}, "
                   f"min_consensus_strength={min_consensus_strength}")
    
    def load_episode_raw_data(self, episode_date: str, data_dir: str = "training_data/episodes") -> Dict[str, Any]:
        """
        Load raw episode JSON data for accessing risk management agent info.
        
        Args:
            episode_date: Episode date in YYYY-MM-DD format
            data_dir: Directory containing episode files
            
        Returns:
            Raw episode data dictionary
        """
        if episode_date in self._episode_data_cache:
            return self._episode_data_cache[episode_date]
        
        episode_file = Path(data_dir) / f"episode_{episode_date}.json"
        
        if episode_file.exists():
            with open(episode_file, 'r') as f:
                raw_data = json.load(f)
            self._episode_data_cache[episode_date] = raw_data
            return raw_data
        else:
            logger.warning(f"Episode file not found: {episode_file}")
            return {}
    
    def get_consensus_tickers_from_episodes(self, episodes: List[TrainingEpisode]) -> Set[str]:
        """
        Extract all tickers that were analyzed by risk management agent (consensus tickers).
        
        Args:
            episodes: List of training episodes
            
        Returns:
            Set of ticker symbols that were analyzed by risk management agent
        """
        consensus_tickers = set()
        ticker_episode_counts = defaultdict(int)
        
        # Extract tickers from risk management agent across all episodes
        for episode in episodes:
            risk_mgmt_tickers = self.get_risk_management_tickers_for_episode(episode)
            for ticker in risk_mgmt_tickers:
                consensus_tickers.add(ticker)
                ticker_episode_counts[ticker] += 1
        
        # Filter tickers that appeared in enough episodes
        filtered_tickers = {
            ticker for ticker, count in ticker_episode_counts.items() 
            if count >= self.min_episodes_per_ticker
        }
        
        logger.info(f"Found {len(consensus_tickers)} total consensus tickers, "
                   f"{len(filtered_tickers)} with >= {self.min_episodes_per_ticker} episodes")
        
        return filtered_tickers
    
    def get_risk_management_tickers_for_episode(self, episode: TrainingEpisode) -> Set[str]:
        """
        Get tickers that were analyzed by the risk management agent for this episode.
        
        Args:
            episode: Training episode
            
        Returns:
            Set of tickers analyzed by risk management agent
        """
        risk_mgmt_tickers = set()
        
        # Load raw episode data to access risk_management_agent
        raw_data = self.load_episode_raw_data(episode.date)
        
        if raw_data and 'agent_signals' in raw_data:
            agent_signals = raw_data['agent_signals']
            
            # Look for risk_management_agent
            if 'risk_management_agent' in agent_signals:
                risk_mgmt_data = agent_signals['risk_management_agent']
                
                # Extract tickers from risk management agent data
                for ticker, ticker_data in risk_mgmt_data.items():
                    # Skip any non-ticker entries (like metadata)
                    if not ticker.startswith('_') and isinstance(ticker_data, dict):
                        risk_mgmt_tickers.add(ticker)
        
        
        return risk_mgmt_tickers
    
    def get_consensus_tickers_for_episode(self, episode: TrainingEpisode) -> Set[str]:
        """
        Get consensus tickers for a specific episode (same as risk management tickers).
        
        Args:
            episode: Training episode
            
        Returns:
            Set of consensus tickers (from risk management agent)
        """
        return self.get_risk_management_tickers_for_episode(episode)
    
    def calculate_ticker_return(
        self, 
        current_episode: TrainingEpisode, 
        next_episode: TrainingEpisode, 
        ticker: str
    ) -> Optional[float]:
        """
        Calculate the return achieved by holding ticker from current to next episode.
        
        Args:
            current_episode: Current episode (end of day T)
            next_episode: Next episode (start of day T+1)
            ticker: Ticker symbol
            
        Returns:
            Return achieved by holding ticker, or None if no position
        """
        try:
            # Get close prices (evaluation_prices) from both episodes
            current_market_data = current_episode.market_data
            next_market_data = next_episode.market_data
            
            current_price = None
            next_price = None
            
            # Get current close price
            if isinstance(current_market_data, dict) and 'evaluation_prices' in current_market_data:
                current_price = current_market_data['evaluation_prices'].get(ticker)
            
            # Get next close price  
            if isinstance(next_market_data, dict) and 'evaluation_prices' in next_market_data:
                next_price = next_market_data['evaluation_prices'].get(ticker)
            
            if current_price is None or next_price is None:
                return None
            
            # Calculate close-to-close return
            return (next_price - current_price) / current_price
                
        except Exception as e:
            logger.debug(f"Error calculating return for {ticker}: {e}")
            return None
    
    def extract_pretrade_features(self, episode: TrainingEpisode, ticker: str) -> Dict[str, float]:
        """
        Extract features using only pre-trade information to avoid look-ahead bias.
        
        Args:
            episode: Training episode
            ticker: Ticker symbol
            
        Returns:
            Dictionary of features for the ticker
        """
        features = {}
        
        # 1. Analyst signals for this ticker (available before trading)
        for agent_name, agent_signals in episode.agent_signals.items():
            if agent_name in ['forward_looking_agent', 'macro_agent']:
                continue
                
            if ticker in agent_signals:
                signal_data = agent_signals[ticker]
                if isinstance(signal_data, dict):
                    signal = signal_data.get('signal', 'neutral')
                    confidence = signal_data.get('confidence', 0)
                    
                    features[f'{agent_name}_signal'] = self.signal_mapping.get(signal, 0.0)
                    features[f'{agent_name}_confidence'] = confidence / 100.0  # Normalize
                else:
                    features[f'{agent_name}_signal'] = 0.0
                    features[f'{agent_name}_confidence'] = 0.0
            else:
                features[f'{agent_name}_signal'] = 0.0
                features[f'{agent_name}_confidence'] = 0.0
        
        # 2. Pre-trade portfolio state for this ticker
        portfolio_before = episode.portfolio_before
        ticker_pos = portfolio_before.positions.get(ticker)
        
        if ticker_pos:
            features['position_long_before'] = ticker_pos.get('long', 0.0)
            features['position_short_before'] = ticker_pos.get('short', 0.0)
            features['position_net_before'] = ticker_pos.get('long', 0.0) - ticker_pos.get('short', 0.0)
            features['long_cost_basis'] = ticker_pos.get('long_cost_basis', 0.0)
            features['short_cost_basis'] = ticker_pos.get('short_cost_basis', 0.0)
        else:
            features['position_long_before'] = 0.0
            features['position_short_before'] = 0.0
            features['position_net_before'] = 0.0
            features['long_cost_basis'] = 0.0
            features['short_cost_basis'] = 0.0
        
        # 3. Market data (prices available before trading)
        market_data = episode.market_data
        if isinstance(market_data, dict) and 'execution_prices' in market_data and ticker in market_data['execution_prices']:
            features['execution_price'] = market_data['execution_prices'][ticker]
        else:
            features['execution_price'] = 0.0
        
        # 4. Portfolio-level features (pre-trade)
        features['portfolio_cash_before'] = portfolio_before.cash
        features['portfolio_margin_requirement'] = portfolio_before.margin_requirement
        
        # 5. Market exposure features (pre-trade)
        if isinstance(market_data, dict):
            features['long_exposure_before'] = market_data.get('long_exposure', 0.0)
            features['short_exposure_before'] = market_data.get('short_exposure', 0.0)
            features['gross_exposure_before'] = market_data.get('gross_exposure', 0.0)
            features['net_exposure_before'] = market_data.get('net_exposure', 0.0)
        else:
            features['long_exposure_before'] = 0.0
            features['short_exposure_before'] = 0.0
            features['gross_exposure_before'] = 0.0
            features['net_exposure_before'] = 0.0
        
        return features
    
    def create_ticker_training_data(self, episodes: List[TrainingEpisode]) -> List[Dict[str, Any]]:
        """
        Create training data for individual ticker return prediction.
        
        Args:
            episodes: List of training episodes (must be sorted by date)
            
        Returns:
            List of training samples, each containing features, target return, ticker, and date
        """
        training_data = []
        
        # Get historical consensus tickers
        self.historical_consensus_tickers = self.get_consensus_tickers_from_episodes(episodes)
        logger.info(f"Using {len(self.historical_consensus_tickers)} historical consensus tickers")
        
        # Create training samples
        for i in range(len(episodes) - 1):  # -1 because we need next episode for targets
            current_episode = episodes[i]
            next_episode = episodes[i + 1]
            
            # Get consensus tickers for current episode
            consensus_tickers = self.get_consensus_tickers_for_episode(current_episode)
            
            # Only include tickers that are both consensus now AND historical consensus
            valid_tickers = consensus_tickers.intersection(self.historical_consensus_tickers)
            
            for ticker in valid_tickers:
                # Extract pre-trade features
                features = self.extract_pretrade_features(current_episode, ticker)
                
                # Calculate target return (next episode)
                target_return = self.calculate_ticker_return(current_episode, next_episode, ticker)
                
                if target_return is not None:
                    training_data.append({
                        'features': features,
                        'target': target_return,
                        'ticker': ticker,
                        'date': current_episode.date,
                        'next_date': next_episode.date
                    })
        
        logger.info(f"Created {len(training_data)} ticker training samples")
        return training_data
    
    def fit(self, episodes: List[TrainingEpisode]) -> 'TickerPreprocessor':
        """
        Fit the preprocessor on training episodes.
        
        Args:
            episodes: List of training episodes (sorted by date)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting TickerPreprocessor on {len(episodes)} episodes")
        
        # Create ticker training data
        training_data = self.create_ticker_training_data(episodes)
        
        if not training_data:
            raise ValueError("No training data created - check episode data and consensus criteria")
        
        # Extract all features into a DataFrame
        features_list = []
        targets = []
        
        for sample in training_data:
            features_list.append(sample['features'])
            targets.append(sample['target'])
        
        features_df = pd.DataFrame(features_list)
        targets_array = np.array(targets)
        
        # Store feature names
        self.feature_names = list(features_df.columns)
        
        # Create and fit scalers
        self.scalers = create_scalers(
            features_df,
            feature_scaler=self.feature_scaler_type,
            target_scaler=self.target_scaler_type
        )
        
        self.is_fitted = True
        logger.info(f"TickerPreprocessor fitted with {len(self.feature_names)} features")
        
        return self
    
    def transform(self, episodes: List[TrainingEpisode]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Transform episodes into training sequences.
        
        Args:
            episodes: List of training episodes
            
        Returns:
            Tuple of (X, y, metadata) where:
            - X: Input sequences for LSTM [num_sequences, sequence_length, num_features]
            - y: Target returns [num_sequences, prediction_horizon]
            - metadata: List of metadata for each sequence
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Create ticker training data
        training_data = self.create_ticker_training_data(episodes)
        
        if not training_data:
            logger.warning("No training data created during transform")
            return np.array([]), np.array([]), []
        
        # Group by ticker for sequence creation
        ticker_data = defaultdict(list)
        for sample in training_data:
            ticker_data[sample['ticker']].append(sample)
        
        # Create sequences
        sequences = []
        targets = []
        metadata = []
        
        for ticker, ticker_samples in ticker_data.items():
            if len(ticker_samples) < self.sequence_length:
                continue  # Not enough data for this ticker
            
            # Sort by date to ensure proper temporal ordering
            ticker_samples.sort(key=lambda x: x['date'])
            
            # Extract features and targets
            ticker_features = []
            ticker_targets = []
            ticker_dates = []
            
            for sample in ticker_samples:
                features_dict = sample['features']
                # Ensure all expected features are present
                feature_vector = []
                for feature_name in self.feature_names:
                    feature_vector.append(features_dict.get(feature_name, 0.0))
                
                ticker_features.append(feature_vector)
                ticker_targets.append(sample['target'])
                ticker_dates.append(sample['date'])
            
            # Convert to arrays
            ticker_features = np.array(ticker_features)
            ticker_targets = np.array(ticker_targets)
            
            # Convert to DataFrame for scaling
            ticker_features_df = pd.DataFrame(ticker_features, columns=self.feature_names)
            
            # Apply scaling
            ticker_features_scaled_df = apply_scalers(
                ticker_features_df, 
                self.scalers
            )
            
            # Convert back to array
            ticker_features_scaled = ticker_features_scaled_df.values
            
            # Create sequences for this ticker
            for i in range(len(ticker_features_scaled) - self.sequence_length + 1):
                # Input sequence: i to i+sequence_length-1
                seq_features = ticker_features_scaled[i:i + self.sequence_length]
                
                # Target: the return at position i+sequence_length-1 (last day of sequence)
                seq_target = ticker_targets[i + self.sequence_length - 1]
                
                sequences.append(seq_features)
                targets.append(seq_target)
                metadata.append({
                    'ticker': ticker,
                    'start_date': ticker_dates[i],
                    'end_date': ticker_dates[i + self.sequence_length - 1],
                    'sequence_index': i,
                    'ticker_sequence_length': len(ticker_samples)
                })
        
        if not sequences:
            logger.warning("No sequences created")
            return np.array([]), np.array([]), []
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y = np.array(targets).reshape(-1, 1)
        
        # Scale targets
        # Use unscaled targets (returns are already in reasonable scale)        
        logger.info(f"Created {len(sequences)} sequences from {len(ticker_data)} tickers")
        logger.info(f"Input shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y.flatten(), metadata
    
    def fit_transform(self, episodes: List[TrainingEpisode]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Fit the preprocessor and transform episodes in one step.
        
        Args:
            episodes: List of training episodes
            
        Returns:
            Transformed data as (X, y, metadata)
        """
        return self.fit(episodes).transform(episodes)
    
    def save(self, filepath: str):
        """Save the fitted preprocessor to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        save_data = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'feature_scaler_type': self.feature_scaler_type,
            'target_scaler_type': self.target_scaler_type,
            'min_consensus_strength': self.min_consensus_strength,
            'min_episodes_per_ticker': self.min_episodes_per_ticker,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'historical_consensus_tickers': list(self.historical_consensus_tickers),
            'signal_mapping': self.signal_mapping,
            'action_mapping': self.action_mapping,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"TickerPreprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TickerPreprocessor':
        """Load a fitted preprocessor from disk."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create instance
        preprocessor = cls(
            sequence_length=save_data['sequence_length'],
            prediction_horizon=save_data['prediction_horizon'],
            feature_scaler=save_data['feature_scaler_type'],
            target_scaler=save_data['target_scaler_type'],
            min_consensus_strength=save_data['min_consensus_strength'],
            min_episodes_per_ticker=save_data['min_episodes_per_ticker']
        )
        
        # Restore fitted state
        preprocessor.scalers = save_data['scalers']
        preprocessor.feature_names = save_data['feature_names']
        preprocessor.historical_consensus_tickers = set(save_data['historical_consensus_tickers'])
        preprocessor.signal_mapping = save_data['signal_mapping']
        preprocessor.action_mapping = save_data['action_mapping']
        preprocessor.is_fitted = save_data['is_fitted']
        
        logger.info(f"TickerPreprocessor loaded from {filepath}")
        return preprocessor
    
    def get_valid_tickers_for_prediction(self, episode: TrainingEpisode) -> Set[str]:
        """
        Get tickers that are valid for prediction in the given episode.
        
        Args:
            episode: Current episode
            
        Returns:
            Set of tickers that can be predicted (consensus now + historical consensus)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before getting valid tickers")
        
        # Get current consensus tickers
        current_consensus = self.get_consensus_tickers_for_episode(episode)
        
        # Return intersection with historical consensus
        return current_consensus.intersection(self.historical_consensus_tickers)