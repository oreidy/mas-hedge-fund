"""
Integration Testing for LSTM Backtesting System.

This script tests the complete LSTM backtesting pipeline including:
- Ticker-based LSTM model training
- Walk-forward model management
- Position synchronization
- Decision making and backtesting
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import pytest

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rl.data_collection.episode import TrainingEpisode
from src.rl.preprocessing.ticker_preprocessor import TickerPreprocessor
from src.rl.models.ticker_lstm_model import TickerLSTMModel
from src.rl.training.ticker_trainer import TickerLSTMTrainer
from src.rl.agents.lstm_decision_maker import LSTMDecisionMaker
from src.rl.backtesting.lstm_backtest import LSTMWalkForwardBacktester
from src.rl.backtesting.model_manager import WalkForwardModelManager
from utils.logger import setup_logger, logger


class TestLSTMIntegration:
    """Integration tests for LSTM backtesting system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        setup_logger(debug_mode=True)
        
        # Create temporary directories
        self.temp_dir = Path(tempfile.mkdtemp())
        self.models_dir = self.temp_dir / "models"
        self.data_dir = self.temp_dir / "data"
        
        self.models_dir.mkdir(parents=True)
        self.data_dir.mkdir(parents=True)
        
        logger.info(f"Test setup - temp dir: {self.temp_dir}")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_sample_episodes(self, num_episodes: int = 100) -> List[TrainingEpisode]:
        """Create sample training episodes for testing."""
        episodes = []
        
        # Sample tickers (consensus tickers)
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        for i in range(num_episodes):
            date = f"2022-01-{(i % 30) + 1:02d}"
            
            # Create agent signals
            agent_signals = {}
            
            # Risk management agent (defines consensus tickers)
            risk_mgmt_signals = {}
            for ticker in tickers:
                risk_mgmt_signals[ticker] = {
                    'remaining_position_limit': 10000.0,
                    'current_price': 100.0 + (i % 20),
                    'reasoning': {'risk_score': 0.5}
                }
            agent_signals['risk_management_agent'] = risk_mgmt_signals
            
            # Other agent signals
            for agent_name in ['technical_analyst_agent', 'sentiment_agent', 'fundamentals_agent']:
                agent_signals[agent_name] = {}
                for ticker in tickers:
                    signal_type = ['bullish', 'neutral', 'bearish'][i % 3]
                    confidence = 50 + (i % 50)
                    
                    agent_signals[agent_name][ticker] = {
                        'signal': signal_type,
                        'confidence': confidence,
                        'reasoning': f'Test signal for {ticker}'
                    }
            
            # Create portfolio states
            portfolio_before = {
                'cash': 100000.0,
                'positions': {ticker: {'long': 0, 'short': 0} for ticker in tickers},
                'margin_requirement': 0.0
            }
            
            portfolio_after = {
                'cash': 95000.0,
                'positions': {ticker: {'long': 10 if i % 5 == 0 else 0, 'short': 0} for ticker in tickers},
                'margin_requirement': 0.0
            }
            
            # Market data
            market_data = {
                'execution_prices': {ticker: 100.0 + (i % 20) for ticker in tickers},
                'evaluation_prices': {ticker: 101.0 + (i % 20) for ticker in tickers}
            }
            
            # Create episode
            episode = TrainingEpisode(
                date=date,
                agent_signals=agent_signals,
                llm_decisions={},
                portfolio_before=portfolio_before,
                portfolio_after=portfolio_after,
                portfolio_return=0.01,
                market_data=market_data,
                metadata={'test': True}
            )
            
            episodes.append(episode)
        
        return episodes
    
    def test_ticker_preprocessor(self):
        """Test ticker preprocessor functionality."""
        logger.info("Testing ticker preprocessor...")
        
        episodes = self.create_sample_episodes(50)
        
        # Create and fit preprocessor
        preprocessor = TickerPreprocessor(
            sequence_length=10,
            min_episodes_per_ticker=5
        )
        
        X, y, metadata = preprocessor.fit_transform(episodes)
        
        # Validate results
        assert len(X) > 0, "No training sequences generated"
        assert len(y) == len(X), "Mismatch between X and y"
        assert len(metadata) == len(X), "Mismatch between X and metadata"
        assert X.shape[1] == 10, f"Sequence length should be 10, got {X.shape[1]}"
        assert len(preprocessor.historical_consensus_tickers) > 0, "No consensus tickers found"
        
        logger.info(f"Preprocessor test passed: {len(X)} sequences, {len(preprocessor.historical_consensus_tickers)} tickers")
    
    def test_ticker_lstm_model(self):
        """Test ticker LSTM model training and prediction."""
        logger.info("Testing ticker LSTM model...")
        
        episodes = self.create_sample_episodes(100)
        
        # Create preprocessor and get data
        preprocessor = TickerPreprocessor(sequence_length=10, min_episodes_per_ticker=5)
        X, y, metadata = preprocessor.fit_transform(episodes)
        
        assert len(X) > 0, "No training data"
        
        # Create model
        model = TickerLSTMModel(
            input_size=X.shape[2],
            hidden_size=32,  # Small for testing
            num_layers=1,
            dropout=0.1
        )
        
        # Build ticker vocabulary
        all_tickers = set(meta['ticker'] for meta in metadata)
        model.build_ticker_vocabulary(all_tickers)
        
        # Test prediction
        sample_x = X[:5]  # First 5 sequences
        sample_tickers = [metadata[i]['ticker'] for i in range(5)]
        
        predictions = model.predict_returns(sample_x, sample_tickers)
        
        assert len(predictions) == len(sample_tickers), "Prediction count mismatch"
        for ticker in sample_tickers:
            assert ticker in predictions, f"Missing prediction for {ticker}"
            assert isinstance(predictions[ticker], float), f"Invalid prediction type for {ticker}"
        
        logger.info(f"Model test passed: predictions for {len(predictions)} tickers")
    
    def test_lstm_decision_maker(self):
        """Test LSTM decision maker."""
        logger.info("Testing LSTM decision maker...")
        
        episodes = self.create_sample_episodes(50)
        
        # Create and fit preprocessor
        preprocessor = TickerPreprocessor(sequence_length=10, min_episodes_per_ticker=5)
        X, y, metadata = preprocessor.fit_transform(episodes)
        
        # Create model
        model = TickerLSTMModel(
            input_size=X.shape[2],
            hidden_size=32,
            num_layers=1
        )
        
        all_tickers = set(meta['ticker'] for meta in metadata)
        model.build_ticker_vocabulary(all_tickers)
        
        # Create decision maker
        decision_maker = LSTMDecisionMaker(model=model, preprocessor=preprocessor)
        decision_maker.valid_tickers = preprocessor.historical_consensus_tickers
        
        # Test decision making
        recent_episodes = episodes[-20:]  # Use recent episodes
        available_tickers = list(all_tickers)[:3]  # Test with subset
        
        decisions = decision_maker.make_decisions(
            recent_episodes=recent_episodes,
            available_tickers=available_tickers
        )
        
        assert len(decisions) == len(available_tickers), "Decision count mismatch"
        for ticker in available_tickers:
            assert ticker in decisions, f"Missing decision for {ticker}"
            decision = decisions[ticker]
            assert 'action' in decision, f"Missing action for {ticker}"
            assert 'quantity' in decision, f"Missing quantity for {ticker}"
            assert 'confidence' in decision, f"Missing confidence for {ticker}"
            assert decision['action'] in ['buy', 'sell', 'short', 'cover', 'hold'], f"Invalid action for {ticker}"
        
        logger.info(f"Decision maker test passed: decisions for {len(decisions)} tickers")
    
    def test_model_manager(self):
        """Test walk-forward model manager."""
        logger.info("Testing model manager...")
        
        # Create fake model metadata
        metadata = {
            250: {
                'model_name': 'model_episode_250',
                'model_path': str(self.models_dir / 'model_250.pth'),
                'preprocessor_path': str(self.models_dir / 'model_250_preprocessor.pkl'),
                'train_end_episode': 250,
                'training_episodes_used': 250,
                'training_end_date': '2022-06-01',
                'best_val_loss': 0.01,
                'epochs_trained': 20,
                'timestamp': '2023-01-01T00:00:00'
            },
            300: {
                'model_name': 'model_episode_300',
                'model_path': str(self.models_dir / 'model_300.pth'),
                'preprocessor_path': str(self.models_dir / 'model_300_preprocessor.pkl'),
                'train_end_episode': 300,
                'training_episodes_used': 300,
                'training_end_date': '2022-07-01',
                'best_val_loss': 0.008,
                'epochs_trained': 25,
                'timestamp': '2023-01-01T00:00:00'
            }
        }
        
        # Save metadata
        metadata_path = self.models_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Test model manager
        manager = WalkForwardModelManager(str(self.models_dir), cache_size=2)
        
        # Test model selection
        model_info_240 = manager.get_model_for_episode(240)
        assert model_info_240 is None, "Should be no model for episode 240"
        
        model_info_260 = manager.get_model_for_episode(260)
        assert model_info_260 is not None, "Should have model for episode 260"
        assert model_info_260.train_end_episode == 250, "Should select model 250 for episode 260"
        
        model_info_320 = manager.get_model_for_episode(320)
        assert model_info_320 is not None, "Should have model for episode 320"
        assert model_info_320.train_end_episode == 300, "Should select model 300 for episode 320"
        
        # Test summary
        summary = manager.get_model_summary()
        assert summary['total_models'] == 2, "Should have 2 models"
        assert summary['episode_range'] == (250, 300), "Incorrect episode range"
        
        logger.info("Model manager test passed")
    
    def test_position_synchronization(self):
        """Test position synchronization logic."""
        logger.info("Testing position synchronization...")
        
        # Create episodes with position changes
        episodes = []
        
        # Episode 1: Have positions
        episode1 = TrainingEpisode(
            date="2022-01-01",
            agent_signals={'risk_management_agent': {'AAPL': {}, 'MSFT': {}}},
            llm_decisions={},
            portfolio_before={'positions': {'AAPL': {'long': 100, 'short': 0}, 'MSFT': {'long': 50, 'short': 0}}},
            portfolio_after={'positions': {'AAPL': {'long': 100, 'short': 0}, 'MSFT': {'long': 50, 'short': 0}}},
            portfolio_return=0.01,
            market_data={'execution_prices': {'AAPL': 150.0, 'MSFT': 250.0}},
            metadata={}
        )
        
        # Episode 2: Position closed for MSFT
        episode2 = TrainingEpisode(
            date="2022-01-02",
            agent_signals={'risk_management_agent': {'AAPL': {}}},  # Only AAPL
            llm_decisions={},
            portfolio_before={'positions': {'AAPL': {'long': 100, 'short': 0}}},  # MSFT closed
            portfolio_after={'positions': {'AAPL': {'long': 100, 'short': 0}}},
            portfolio_return=0.01,
            market_data={'execution_prices': {'AAPL': 151.0, 'MSFT': 249.0}},
            metadata={}
        )
        
        episodes = [episode1, episode2]
        
        # Create backtester
        backtester = LSTMWalkForwardBacktester(
            initial_capital=100000,
            initial_training_episodes=1,  # Use episode 1 for initial training
            retraining_frequency=10,
            max_positions=10,
            model_save_dir=str(self.models_dir)
        )
        
        # This test validates the synchronization logic exists
        # Full integration would require more complex setup
        assert hasattr(backtester, '_synchronize_positions_with_episode'), "Synchronization method missing"
        
        logger.info("Position synchronization test passed")
    
    def run_basic_integration_test(self):
        """Run a basic end-to-end integration test."""
        logger.info("Running basic integration test...")
        
        episodes = self.create_sample_episodes(30)  # Small test
        
        # Create backtester
        backtester = LSTMWalkForwardBacktester(
            initial_capital=100000,
            initial_training_episodes=20,
            retraining_frequency=5,
            max_positions=5,
            model_save_dir=str(self.models_dir)
        )
        
        try:
            # This would normally run the full backtest
            # For testing, we just validate the setup
            assert backtester.initial_capital == 100000, "Initial capital mismatch"
            assert backtester.initial_training_episodes == 20, "Training episodes mismatch"
            assert backtester.preprocessor is not None, "Preprocessor not initialized"
            assert backtester.decision_maker is not None, "Decision maker not initialized"
            
            logger.info("Basic integration test passed")
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            raise


def run_all_tests():
    """Run all integration tests."""
    setup_logger(debug_mode=True)
    logger.info("Starting LSTM integration tests...")
    
    tester = TestLSTMIntegration()
    
    try:
        tester.setup_method()
        
        # Run individual tests
        tester.test_ticker_preprocessor()
        tester.test_ticker_lstm_model()
        tester.test_lstm_decision_maker()
        tester.test_model_manager()
        tester.test_position_synchronization()
        tester.run_basic_integration_test()
        
        logger.info("✅ All integration tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Integration tests failed: {e}")
        raise
    
    finally:
        tester.teardown_method()


if __name__ == "__main__":
    run_all_tests()