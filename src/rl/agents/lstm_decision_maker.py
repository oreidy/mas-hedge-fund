"""
LSTM Decision Maker for Trading Actions.

This module contains the LSTMDecisionMaker class that uses ticker-based LSTM models
to make trading decisions in the same format as the LLM equity agent.

Updated for ticker-based LSTM architecture that predicts individual ticker returns.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

from ..data_collection.episode import TrainingEpisode
from ..preprocessing.ticker_preprocessor import TickerPreprocessor
from ..models.ticker_lstm_model import TickerLSTMModel
from utils.logger import logger


class LSTMDecisionMaker:
    """
    High-level LSTM decision maker that processes episodes and generates trading decisions.
    
    This class uses the ticker-based LSTM model to predict individual ticker returns and 
    converts them to trading decisions in the same format as the LLM equity agent.
    
    Updated for new ticker-based architecture.
    """
    
    def __init__(
        self,
        model: Optional[TickerLSTMModel] = None,
        preprocessor: Optional[TickerPreprocessor] = None,
        sequence_length: int = 20
    ):
        """
        Initialize the LSTM decision maker.
        
        Args:
            model: Trained ticker LSTM model (optional, can be loaded later)
            preprocessor: Ticker preprocessor (optional, will create default if None)
            sequence_length: Length of input sequences
        """
        self.model = model
        self.sequence_length = sequence_length
        self.valid_tickers = set()  # Historical consensus tickers from preprocessor
        
        # Initialize preprocessor if not provided
        if preprocessor is None:
            self.preprocessor = TickerPreprocessor(
                sequence_length=sequence_length,
                prediction_horizon=1,
                min_consensus_strength=0.5,
                min_episodes_per_ticker=10
            )
        else:
            self.preprocessor = preprocessor
            # Set valid tickers from preprocessor if it's fitted
            if self.preprocessor.is_fitted:
                self.valid_tickers = self.preprocessor.historical_consensus_tickers
                logger.info(f"Set {len(self.valid_tickers)} valid tickers from fitted preprocessor")
        
        logger.info("LSTMDecisionMaker initialized for ticker-based predictions")
    
    def load_model(self, model_path: str, preprocessor_path: str = None):
        """
        Load a trained ticker LSTM model and preprocessor.
        
        Args:
            model_path: Path to saved TickerLSTMModel
            preprocessor_path: Path to saved TickerPreprocessor (optional)
        """
        try:
            # Load the ticker LSTM model
            self.model = TickerLSTMModel.load(model_path)
            logger.info(f"Ticker LSTM model loaded from {model_path}")
            
            # Load preprocessor if provided
            if preprocessor_path:
                self.preprocessor = TickerPreprocessor.load(preprocessor_path)
                logger.info(f"Ticker preprocessor loaded from {preprocessor_path}")
                
                # Get valid tickers from preprocessor
                if self.preprocessor.is_fitted:
                    self.valid_tickers = self.preprocessor.historical_consensus_tickers
                    logger.info(f"Loaded {len(self.valid_tickers)} historical consensus tickers")
            
            # Log model info
            model_info = self.model.get_model_info()
            logger.info(f"Model configuration: {model_info}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def make_decisions(
        self,
        recent_episodes: List[TrainingEpisode],
        available_tickers: List[str],
        portfolio_state: Dict[str, Any] = None,
        risk_limits: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Make trading decisions for available tickers based on recent episodes.
        
        Args:
            recent_episodes: List of recent episodes (should include current day)
            available_tickers: List of tickers to make decisions for (consensus tickers)
            portfolio_state: Current portfolio state (optional)
            risk_limits: Risk management limits per ticker (optional)
            
        Returns:
            Dictionary mapping ticker to decision (action, quantity, confidence, reasoning)
        """
        if self.model is None:
            raise ValueError("No ticker LSTM model loaded. Call load_model() first.")
        
        if not self.preprocessor.is_fitted:
            raise ValueError("Preprocessor not fitted. Load a fitted preprocessor.")
        
        if len(recent_episodes) == 0:
            logger.warning("No episodes provided for prediction")
            return {}
        
        # Filter available tickers to only those in our valid ticker set AND model vocabulary
        valid_available_tickers = []
        for ticker in available_tickers:
            if ticker in self.valid_tickers and ticker in self.model.ticker_vocab:
                valid_available_tickers.append(ticker)
            elif ticker not in self.valid_tickers:
                logger.debug(f"Ticker {ticker} not in valid historical tickers")
            elif ticker not in self.model.ticker_vocab:
                logger.debug(f"Ticker {ticker} not in model vocabulary")
        
        if not valid_available_tickers:
            logger.info(f"No valid tickers found in available list. Available: {available_tickers[:5]}, Valid: {list(self.valid_tickers)[:5]}")
            return {ticker: self._create_hold_decision(ticker, "Not in LSTM training data") 
                   for ticker in available_tickers}
        
        try:
            # Transform recent episodes to get features
            X, y, metadata = self.preprocessor.transform(recent_episodes)
            
            if len(X) == 0:
                logger.warning("No training data generated from recent episodes")
                return {ticker: self._create_hold_decision(ticker, "No sequence data available") 
                       for ticker in available_tickers}
            
            # Get predictions for valid tickers
            ticker_predictions = {}
            
            # Group by ticker and get most recent sequence for each
            ticker_sequences = {}
            for i, meta in enumerate(metadata):
                ticker = meta['ticker']
                if ticker in valid_available_tickers:
                    if ticker not in ticker_sequences or meta['end_date'] > ticker_sequences[ticker]['end_date']:
                        ticker_sequences[ticker] = {
                            'sequence': X[i:i+1],  # Keep batch dimension
                            'end_date': meta['end_date']
                        }
            
            # Get predictions for each ticker
            for ticker in valid_available_tickers:
                if ticker in ticker_sequences:
                    ticker_data = ticker_sequences[ticker]
                    X_ticker = torch.FloatTensor(ticker_data['sequence']).to(self.model.device)
                    
                    # Get prediction
                    prediction = self.model.predict_returns(X_ticker, [ticker])
                    ticker_predictions[ticker] = prediction[ticker]
                else:
                    logger.debug(f"No sequence data available for ticker {ticker}")
            
            # Convert predictions to decisions
            decisions = self._convert_predictions_to_decisions(
                ticker_predictions, available_tickers, portfolio_state, risk_limits
            )
            
            logger.info(f"LSTM made decisions for {len(ticker_predictions)} tickers")
            if ticker_predictions:
                sample_preds = {k: v for k, v in list(ticker_predictions.items())[:3]}
                logger.debug(f"Sample predictions: {sample_preds}")
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error making LSTM decisions: {e}")
            # Return hold decisions for all tickers as fallback
            return {ticker: self._create_hold_decision(ticker, f"Prediction error: {str(e)}") 
                   for ticker in available_tickers}
    
    def _convert_predictions_to_decisions(
        self, 
        ticker_predictions: Dict[str, float], 
        available_tickers: List[str],
        portfolio_state: Dict[str, Any] = None,
        risk_limits: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert individual ticker return predictions to trading decisions.
        
        Args:
            ticker_predictions: Dictionary mapping ticker to predicted return
            available_tickers: List of all tickers to make decisions for
            portfolio_state: Current portfolio state (optional)
            risk_limits: Risk management limits per ticker (optional)
            
        Returns:
            Dictionary mapping ticker to decision
        """
        ticker_decisions = {}
        
        for ticker in available_tickers:
            if ticker not in ticker_predictions:
                # No prediction available - hold
                ticker_decisions[ticker] = self._create_hold_decision(
                    ticker, "No LSTM prediction available"
                )
                continue
            
            predicted_return = ticker_predictions[ticker]
            abs_return = abs(predicted_return)
            
            # Get risk management limits for this ticker
            position_limit = 0.0
            current_price = 0.0
            if risk_limits and ticker in risk_limits:
                position_limit = risk_limits[ticker].get('remaining_position_limit', 0.0)
                current_price = risk_limits[ticker].get('current_price', 0.0)
            
            # Decision logic based on individual ticker return prediction
            if predicted_return > 0.005:  # Positive return > 0.5%
                action = 'buy'
                confidence = min(0.9, 0.5 + abs_return * 20)  # Scale confidence with magnitude
                
                # Use risk management position limit to calculate quantity
                if position_limit > 0 and current_price > 0:
                    max_shares = int(position_limit / current_price)
                    # Scale quantity based on signal strength, but respect risk limits
                    signal_scaled_shares = int(abs_return * 1000)  # Scale with return magnitude
                    quantity = min(max_shares, max(1, signal_scaled_shares))
                else:
                    quantity = max(1, int(abs_return * 500))  # Fallback logic
                    
            elif predicted_return < -0.005:  # Negative return < -0.5%
                # Check if we have existing position to determine sell vs short
                if portfolio_state and "positions" in portfolio_state and ticker in portfolio_state["positions"]:
                    existing_long = portfolio_state["positions"][ticker].get("long", 0)
                    if existing_long > 0:
                        action = 'sell'  # Close long position
                        # Sell based on signal strength, but don't exceed holdings
                        signal_scaled_shares = int(abs_return * 1000)
                        quantity = min(existing_long, max(1, signal_scaled_shares))
                    else:
                        action = 'short'  # Open short position
                        # Use risk limits for short positions too
                        if position_limit > 0 and current_price > 0:
                            max_shares = int(position_limit / current_price)
                            signal_scaled_shares = int(abs_return * 1000)
                            quantity = min(max_shares, max(1, signal_scaled_shares))
                        else:
                            quantity = max(1, int(abs_return * 500))
                else:
                    # No portfolio state or no existing position - default to modest short
                    action = 'short'
                    if position_limit > 0 and current_price > 0:
                        max_shares = int(position_limit / current_price)
                        quantity = min(max_shares, max(1, int(abs_return * 500)))
                    else:
                        quantity = max(1, int(abs_return * 300))  # More conservative without risk limits
                
                confidence = min(0.9, 0.5 + abs_return * 20)
            else:  # Weak signal (-0.5% to +0.5%)
                action = 'hold'
                confidence = 0.5
                quantity = 0
            
            # Create reasoning including risk management info
            reasoning = f"LSTM predicted return: {predicted_return:.4f} ({predicted_return*100:.2f}%)"
            if risk_limits and ticker in risk_limits:
                reasoning += f", Risk limit: ${position_limit:.0f}"
            
            ticker_decisions[ticker] = {
                'action': action,
                'quantity': quantity,
                'confidence': confidence,
                'reasoning': reasoning
            }
        
        # Log summary
        actions_summary = {}
        for decision in ticker_decisions.values():
            action = decision['action']
            actions_summary[action] = actions_summary.get(action, 0) + 1
        
        logger.info(f"LSTM decisions summary: {actions_summary}")
        
        return ticker_decisions
    
    def _create_hold_decision(self, ticker: str, reason: str) -> Dict[str, Any]:
        """Create a hold decision with given reason."""
        return {
            'action': 'hold',
            'quantity': 0,
            'confidence': 0.5,
            'reasoning': f"{ticker}: {reason}"
        }
    
    def get_valid_tickers(self) -> Set[str]:
        """Get the set of valid tickers that can be predicted."""
        return self.valid_tickers.copy()
    
    def is_ticker_valid(self, ticker: str) -> bool:
        """Check if a ticker can be predicted by this model."""
        return ticker in self.valid_tickers
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"model_loaded": False}
        
        model_info = self.model.get_model_info()
        model_info.update({
            "valid_tickers_count": len(self.valid_tickers),
            "preprocessor_fitted": self.preprocessor.is_fitted,
            "sequence_length": self.sequence_length
        })
        
        return model_info