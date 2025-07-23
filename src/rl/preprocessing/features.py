"""
Feature extraction utilities for RL training.

This module provides utilities for extracting and engineering features
from trading episodes that are useful for LSTM training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from scipy import stats

from ..data_collection.episode import TrainingEpisode
from utils.logger import logger


class FeatureExtractor:
    """
    Extracts and engineers features from trading episodes.
    
    This class provides methods to extract various types of features:
    1. Technical indicators
    2. Signal momentum features
    3. Portfolio composition features
    4. Risk metrics
    5. Market regime features
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.signal_mapping = {
            'bullish': 1.0,
            'neutral': 0.0,
            'bearish': -1.0
        }
        
        self.action_mapping = {
            'buy': 1.0,
            'sell': -1.0,
            'short': -2.0,
            'cover': 2.0,
            'hold': 0.0
        }
    
    def extract_technical_features(self, episodes: List[TrainingEpisode], window_size: int = 5) -> pd.DataFrame:
        """
        Extract technical indicator features from episodes.
        
        Args:
            episodes: List of training episodes
            window_size: Window size for rolling calculations
            
        Returns:
            DataFrame with technical features
        """
        features = []
        
        # Sort episodes by date
        episodes = sorted(episodes, key=lambda x: x.date)
        
        for i, episode in enumerate(episodes):
            row = {'date': episode.date}
            
            # Get historical episodes for window calculations
            start_idx = max(0, i - window_size + 1)
            window_episodes = episodes[start_idx:i + 1]
            
            # Portfolio return momentum
            returns = [ep.portfolio_return for ep in window_episodes]
            row['return_ma'] = np.mean(returns)
            row['return_std'] = np.std(returns)
            row['return_momentum'] = returns[-1] - returns[0] if len(returns) > 1 else 0
            
            # Signal momentum
            signal_momentum = self._calculate_signal_momentum(window_episodes)
            row.update(signal_momentum)
            
            # Decision consistency
            decision_consistency = self._calculate_decision_consistency(window_episodes)
            row.update(decision_consistency)
            
            # Portfolio value trend
            portfolio_trend = self._calculate_portfolio_trend(window_episodes)
            row.update(portfolio_trend)
            
            features.append(row)
        
        return pd.DataFrame(features)
    
    def _calculate_signal_momentum(self, episodes: List[TrainingEpisode]) -> Dict[str, float]:
        """Calculate signal momentum features."""
        if len(episodes) < 2:
            return {}
        
        features = {}
        
        # Track signal changes over time
        agent_signals = {}
        
        for episode in episodes:
            for agent_name, ticker_signals in episode.agent_signals.items():
                if agent_name not in agent_signals:
                    agent_signals[agent_name] = []
                
                # Calculate average signal for this agent on this day
                signals = [self.signal_mapping.get(sig.signal, 0.0) for sig in ticker_signals.values()]
                avg_signal = np.mean(signals) if signals else 0.0
                agent_signals[agent_name].append(avg_signal)
        
        # Calculate momentum for each agent
        for agent_name, signals in agent_signals.items():
            if len(signals) > 1:
                # Signal trend
                trend = np.polyfit(range(len(signals)), signals, 1)[0]
                features[f'{agent_name}_signal_trend'] = trend
                
                # Signal volatility
                features[f'{agent_name}_signal_volatility'] = np.std(signals)
                
                # Signal reversals
                reversals = sum(1 for i in range(1, len(signals)) 
                              if signals[i] * signals[i-1] < 0)
                features[f'{agent_name}_signal_reversals'] = reversals
        
        return features
    
    def _calculate_decision_consistency(self, episodes: List[TrainingEpisode]) -> Dict[str, float]:
        """Calculate decision consistency features."""
        if len(episodes) < 2:
            return {}
        
        features = {}
        
        # Track decision patterns
        actions = []
        quantities = []
        confidences = []
        
        for episode in episodes:
            episode_actions = []
            episode_quantities = []
            episode_confidences = []
            
            for decision in episode.llm_decisions.values():
                action_numeric = self.action_mapping.get(decision.action, 0.0)
                episode_actions.append(action_numeric)
                episode_quantities.append(decision.quantity)
                episode_confidences.append(decision.confidence)
            
            if episode_actions:
                actions.append(np.mean(episode_actions))
                quantities.append(np.mean(episode_quantities))
                confidences.append(np.mean(episode_confidences))
        
        if len(actions) > 1:
            # Action consistency
            features['action_consistency'] = 1.0 - np.std(actions) / (np.abs(np.mean(actions)) + 1e-6)
            
            # Quantity consistency
            features['quantity_consistency'] = 1.0 - np.std(quantities) / (np.mean(quantities) + 1e-6)
            
            # Confidence trend
            features['confidence_trend'] = np.polyfit(range(len(confidences)), confidences, 1)[0]
            
            # Decision volatility
            features['decision_volatility'] = np.std(actions)
        
        return features
    
    def _calculate_portfolio_trend(self, episodes: List[TrainingEpisode]) -> Dict[str, float]:
        """Calculate portfolio trend features."""
        if len(episodes) < 2:
            return {}
        
        features = {}
        
        # Portfolio values
        cash_values = [ep.portfolio_before.cash for ep in episodes]
        
        # Cash trend
        if len(cash_values) > 1:
            features['cash_trend'] = np.polyfit(range(len(cash_values)), cash_values, 1)[0]
            features['cash_volatility'] = np.std(cash_values)
        
        # Position counts
        long_positions = []
        short_positions = []
        
        for episode in episodes:
            long_count = sum(pos['long'] for pos in episode.portfolio_before.positions.values())
            short_count = sum(pos['short'] for pos in episode.portfolio_before.positions.values())
            long_positions.append(long_count)
            short_positions.append(short_count)
        
        if len(long_positions) > 1:
            features['long_position_trend'] = np.polyfit(range(len(long_positions)), long_positions, 1)[0]
            features['short_position_trend'] = np.polyfit(range(len(short_positions)), short_positions, 1)[0]
        
        return features
    
    def extract_risk_features(self, episodes: List[TrainingEpisode], lookback_window: int = 20) -> pd.DataFrame:
        """
        Extract risk-related features from episodes.
        
        Args:
            episodes: List of training episodes
            lookback_window: Window for risk calculations
            
        Returns:
            DataFrame with risk features
        """
        features = []
        
        # Sort episodes by date
        episodes = sorted(episodes, key=lambda x: x.date)
        
        for i, episode in enumerate(episodes):
            row = {'date': episode.date}
            
            # Get historical episodes for window calculations
            start_idx = max(0, i - lookback_window + 1)
            window_episodes = episodes[start_idx:i + 1]
            
            # Return-based risk metrics
            returns = [ep.portfolio_return for ep in window_episodes]
            
            if len(returns) > 1:
                # Volatility
                row['volatility'] = np.std(returns)
                
                # Downside deviation
                downside_returns = [r for r in returns if r < 0]
                row['downside_deviation'] = np.std(downside_returns) if downside_returns else 0.0
                
                # Maximum drawdown
                cumulative_returns = np.cumprod([1 + r for r in returns])
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                row['max_drawdown'] = np.min(drawdowns)
                
                # Sharpe ratio (assuming zero risk-free rate)
                row['sharpe_ratio'] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
                
                # Sortino ratio
                row['sortino_ratio'] = (np.mean(returns) / row['downside_deviation'] 
                                      if row['downside_deviation'] > 0 else 0.0)
                
                # Skewness and kurtosis
                row['return_skewness'] = stats.skew(returns)
                row['return_kurtosis'] = stats.kurtosis(returns)
                
                # Value at Risk (5% VaR)
                row['var_5pct'] = np.percentile(returns, 5)
                
                # Expected shortfall (Conditional VaR)
                var_threshold = row['var_5pct']
                tail_returns = [r for r in returns if r <= var_threshold]
                row['expected_shortfall'] = np.mean(tail_returns) if tail_returns else 0.0
            
            # Position concentration risk
            concentration_risk = self._calculate_concentration_risk(episode)
            row.update(concentration_risk)
            
            features.append(row)
        
        return pd.DataFrame(features)
    
    def _calculate_concentration_risk(self, episode: TrainingEpisode) -> Dict[str, float]:
        """Calculate position concentration risk metrics."""
        features = {}
        
        # Get position values (simplified - using quantities as proxy)
        positions = episode.portfolio_before.positions
        
        long_positions = [pos['long'] for pos in positions.values() if pos['long'] > 0]
        short_positions = [pos['short'] for pos in positions.values() if pos['short'] > 0]
        
        # Long position concentration
        if long_positions:
            total_long = sum(long_positions)
            long_weights = [pos / total_long for pos in long_positions]
            features['long_concentration'] = sum(w**2 for w in long_weights)  # Herfindahl index
            features['long_max_weight'] = max(long_weights)
            features['long_position_count'] = len(long_positions)
        else:
            features['long_concentration'] = 0.0
            features['long_max_weight'] = 0.0
            features['long_position_count'] = 0
        
        # Short position concentration
        if short_positions:
            total_short = sum(short_positions)
            short_weights = [pos / total_short for pos in short_positions]
            features['short_concentration'] = sum(w**2 for w in short_weights)
            features['short_max_weight'] = max(short_weights)
            features['short_position_count'] = len(short_positions)
        else:
            features['short_concentration'] = 0.0
            features['short_max_weight'] = 0.0
            features['short_position_count'] = 0
        
        return features
    
    def extract_market_regime_features(self, episodes: List[TrainingEpisode]) -> pd.DataFrame:
        """
        Extract market regime features from episodes.
        
        Args:
            episodes: List of training episodes
            
        Returns:
            DataFrame with market regime features
        """
        features = []
        
        # Sort episodes by date
        episodes = sorted(episodes, key=lambda x: x.date)
        
        for i, episode in enumerate(episodes):
            row = {'date': episode.date}
            
            # Market data features
            if episode.market_data:
                market_data = episode.market_data
                
                # Exposure ratios
                long_exp = market_data.get('long_exposure', 0)
                short_exp = market_data.get('short_exposure', 0)
                gross_exp = market_data.get('gross_exposure', 0)
                
                row['long_short_ratio'] = long_exp / (short_exp + 1e-6)
                row['net_exposure'] = (long_exp - short_exp) / (gross_exp + 1e-6)
                row['gross_exposure_ratio'] = gross_exp / episode.portfolio_before.cash
            
            # Signal distribution
            signal_dist = self._calculate_signal_distribution(episode)
            row.update(signal_dist)
            
            features.append(row)
        
        return pd.DataFrame(features)
    
    def _calculate_signal_distribution(self, episode: TrainingEpisode) -> Dict[str, float]:
        """Calculate signal distribution features."""
        features = {}
        
        # Collect all signals
        all_signals = []
        for agent_name, ticker_signals in episode.agent_signals.items():
            for signal in ticker_signals.values():
                numeric_signal = self.signal_mapping.get(signal.signal, 0.0)
                all_signals.append(numeric_signal)
        
        if all_signals:
            # Signal distribution
            features['signal_mean'] = np.mean(all_signals)
            features['signal_std'] = np.std(all_signals)
            features['signal_skew'] = stats.skew(all_signals)
            features['signal_kurtosis'] = stats.kurtosis(all_signals)
            
            # Signal regime indicators
            features['bullish_regime'] = 1.0 if np.mean(all_signals) > 0.3 else 0.0
            features['bearish_regime'] = 1.0 if np.mean(all_signals) < -0.3 else 0.0
            features['neutral_regime'] = 1.0 if abs(np.mean(all_signals)) <= 0.3 else 0.0
            
            # Signal dispersion
            features['signal_dispersion'] = np.std(all_signals) / (abs(np.mean(all_signals)) + 1e-6)
        
        return features