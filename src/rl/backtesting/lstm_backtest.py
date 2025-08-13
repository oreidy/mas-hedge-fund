"""
LSTM Walk-Forward Backtesting System.

This module provides walk-forward backtesting for LSTM trading models using
expanding window validation and identical execution logic as the main backtester.
"""

import sys
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from rl.data_collection.episode import TrainingEpisode, AgentSignal, LLMDecision, PortfolioState
from rl.preprocessing.ticker_preprocessor import TickerPreprocessor
from rl.agents.lstm_decision_maker import LSTMDecisionMaker
from rl.backtesting.portfolio_engine import PortfolioEngine
from rl.backtesting.model_manager import WalkForwardModelManager
from utils.display import print_backtest_results, format_backtest_row

logger = logging.getLogger(__name__)


class LSTMWalkForwardBacktester:
    """
    Walk-forward backtesting system for LSTM trading models.
    
    This system implements expanding window validation where:
    1. Initial training on first 250 episodes
    2. Monthly retraining with expanding dataset
    3. Daily trading decisions using current LSTM model
    4. Identical portfolio execution as main backtester
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        initial_training_episodes: int = 250,
        retraining_frequency: int = 30,  # Retrain every 30 days (monthly)
        max_positions: int = 30,
        transaction_cost: float = 0.0005,
        sequence_length: int = 20,
        model_save_dir: str = "src/rl/trained_models"
    ):
        """
        Initialize the walk-forward backtester.
        
        Args:
            initial_capital: Starting portfolio value
            initial_training_episodes: Number of episodes for initial training
            retraining_frequency: How often to retrain the model (in episodes/days)
            max_positions: Maximum number of positions allowed
            transaction_cost: Transaction cost as percentage (0.0005 = 5 basis points)
            sequence_length: LSTM sequence length
            model_save_dir: Directory to save trained models
        """
        self.initial_capital = initial_capital
        self.initial_training_episodes = initial_training_episodes
        self.retraining_frequency = retraining_frequency
        self.max_positions = max_positions
        self.transaction_cost = transaction_cost
        self.sequence_length = sequence_length
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.preprocessor = TickerPreprocessor(
            sequence_length=sequence_length,
            prediction_horizon=1,
            min_consensus_strength=0.5,
            min_episodes_per_ticker=10
        )
        
        self.decision_maker = LSTMDecisionMaker(
            preprocessor=self.preprocessor,
            sequence_length=sequence_length
        )
        
        # Results tracking
        self.results = []
        self.portfolio_values = []
        self.model_performance = []
        
        logger.info(f"LSTMWalkForwardBacktester initialized")
        logger.info(f"Initial training: {initial_training_episodes} episodes")
        logger.info(f"Retraining frequency: {retraining_frequency} episodes")
    
    def extract_consensus_data(self, episode: TrainingEpisode) -> Dict[str, Any]:
        """
        Extract consensus tickers and their risk management limits from an episode.
        
        Args:
            episode: Training episode
            
        Returns:
            Dictionary with consensus_tickers list and risk_limits dict
        """
        risk_mgmt_signals = episode.agent_signals.get('risk_management_agent', {})
        
        consensus_tickers = list(risk_mgmt_signals.keys())
        risk_limits = {}
        
        for ticker in consensus_tickers:
            if ticker in risk_mgmt_signals:
                risk_data = risk_mgmt_signals[ticker]
                # Extract key risk management data
                risk_limits[ticker] = {
                    'remaining_position_limit': risk_data.get('remaining_position_limit', 0.0) * 3.0,
                    'current_price': risk_data.get('current_price', 0.0),
                    'reasoning': risk_data.get('reasoning', {})
                }
        
        return {
            'consensus_tickers': consensus_tickers,
            'risk_limits': risk_limits
        }
    
    def _synchronize_positions_with_episode(
        self, 
        portfolio_engine: PortfolioEngine, 
        episode: TrainingEpisode,
        execution_prices: Dict[str, float]
    ):
        """
        Synchronize portfolio positions with episode data.
        
        The episode data contains positions that were closed in the original LLM run,
        but the LSTM backtester won't know about these closures. This method closes 
        positions that were closed in the original episode data to maintain consistency.
        
        Args:
            portfolio_engine: Portfolio engine to update
            episode: Current episode
            execution_prices: Execution prices for trading
        """
        try:
            # Get positions from episode data (what the original system had)
            episode_positions = set()
            if hasattr(episode, 'portfolio_before') and episode.portfolio_before:
                if hasattr(episode.portfolio_before, 'positions'):
                    episode_positions = set(episode.portfolio_before.positions.keys())
            
            # Get current positions in our portfolio
            current_positions = set()
            for ticker, position in portfolio_engine.portfolio["positions"].items():
                if position["long"] > 0 or position["short"] > 0:
                    current_positions.add(ticker)
            
            # Find positions that were closed in the original episode but we still have
            positions_to_close = current_positions - episode_positions
            
            if positions_to_close:
                logger.info(f"Synchronizing positions: closing {len(positions_to_close)} positions that were closed in original episode")
                
                for ticker in positions_to_close:
                    if ticker in execution_prices:
                        position = portfolio_engine.portfolio["positions"][ticker]
                        
                        # Close long positions
                        if position["long"] > 0:
                            executed_qty = portfolio_engine.execute_trade(
                                ticker, "sell", position["long"], execution_prices[ticker]
                            )
                            logger.debug(f"Synced: Sold {executed_qty} shares of {ticker} (was {position['long']})")
                        
                        # Close short positions
                        if position["short"] > 0:
                            executed_qty = portfolio_engine.execute_trade(
                                ticker, "cover", position["short"], execution_prices[ticker]
                            )
                            logger.debug(f"Synced: Covered {executed_qty} shares of {ticker} (was {position['short']})")
                    else:
                        logger.warning(f"Cannot sync position for {ticker}: no execution price available")
            
        except Exception as e:
            logger.warning(f"Position synchronization failed: {e}")
            # Continue without synchronization - this is not critical
    
    def load_all_episodes(self, episodes_dir: str = "training_data/episodes") -> List[TrainingEpisode]:
        """Load all training episodes in chronological order."""
        episodes_path = Path(episodes_dir)
        json_files = sorted(list(episodes_path.glob("*.json")))
        
        episodes = []
        failed_loads = 0
        
        logger.info(f"Loading {len(json_files)} episodes from {episodes_path}")
        
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
    
    
    def run_walk_forward_backtest(self, episodes: List[TrainingEpisode]) -> Dict[str, Any]:
        """
        Run the complete walk-forward backtest.
        
        Args:
            episodes: All episodes in chronological order
            
        Returns:
            Backtesting results and performance metrics
        """
        if len(episodes) < self.initial_training_episodes + self.sequence_length:
            raise ValueError(f"Need at least {self.initial_training_episodes + self.sequence_length} episodes")
        
        logger.info(f"Starting walk-forward backtest on {len(episodes)} episodes")
        
        # Get ALL tickers that ever appeared in risk management data (for portfolio initialization)
        # But we'll only trade episode-specific tickers to match src/backtester.py behavior
        all_historical_tickers = set()
        for episode in episodes:
            risk_mgmt_signals = episode.agent_signals.get('risk_management_agent', {})
            if risk_mgmt_signals:
                all_historical_tickers.update(risk_mgmt_signals.keys())
        
        all_historical_tickers = sorted(list(all_historical_tickers))
        logger.info(f"Found {len(all_historical_tickers)} historical consensus tickers for portfolio initialization")
        
        # Use historical tickers for portfolio engine initialization (to handle existing positions)
        # Always include bond ETFs even if they don't appear in risk management data
        all_tickers = sorted(list(set(all_historical_tickers + ["SHY", "TLT"])))
        
        # Initialize portfolio engine
        portfolio_engine = PortfolioEngine(
            initial_capital=self.initial_capital,
            all_tickers=all_tickers,
            max_positions=self.max_positions,
            transaction_cost=self.transaction_cost,
            margin_ratio=0.0,  # No margin for simplicity
            bond_etfs=["SHY", "TLT"]
        )
        
        # Initialize model manager for walk-forward
        model_manager = None
        if Path(self.model_save_dir / "model_metadata.json").exists():
            try:
                model_manager = WalkForwardModelManager(
                    models_dir=str(self.model_save_dir),
                    cache_size=3
                )
                logger.info("Using pre-trained walk-forward models")
            except Exception as e:
                logger.warning(f"Could not load walk-forward models: {e}")
                logger.info("Will train models during backtest")
        
        # Track results
        daily_results = []
        table_rows = []  # For displaying daily position tables
        
        # Verify we have pretrained models
        if not model_manager:
            raise ValueError("No pretrained models found. Please run pretrain_models.py first.")
        
        logger.info(f"Using pretrained models from {self.model_save_dir}")
        
        # Walk-forward testing
        current_portfolio_value = self.initial_capital
        
        for i in tqdm(range(self.initial_training_episodes, len(episodes)), desc="Walk-forward backtesting"):
            episode = episodes[i]
            
            # All model selection is handled by model_manager - no retraining needed
            
            # Get pricing data
            execution_prices = episode.market_data.get('execution_prices', {})  # For trade execution (open prices)
            evaluation_prices = episode.market_data.get('evaluation_prices', {})  # For portfolio valuation (close prices)
            
            # Get previous episode's evaluation prices for portfolio_value_before
            previous_evaluation_prices = {}
            if i > 0:
                previous_episode = episodes[i-1]
                previous_evaluation_prices = previous_episode.market_data.get('evaluation_prices', {})
            else:
                # For first episode, use current day's execution prices (open)
                previous_evaluation_prices = execution_prices
            
            # Synchronize positions with episode data to maintain consistency
            self._synchronize_positions_with_episode(
                portfolio_engine, episode, execution_prices
            )
            
            # Get portfolio state before trading
            portfolio_before = portfolio_engine.get_portfolio_state_copy()
            # Value portfolio at previous day's close prices (or first day's open)
            portfolio_value_before = portfolio_engine.get_portfolio_value(previous_evaluation_prices)
            
            # Extract consensus tickers and risk limits for this episode
            consensus_data = self.extract_consensus_data(episode)
            episode_consensus_tickers = consensus_data['consensus_tickers']
            risk_limits = consensus_data['risk_limits']
            
            # Only make decisions for episode-specific consensus tickers that have execution prices
            # This matches src/backtester.py behavior exactly
            execution_prices = episode.market_data.get('execution_prices', {})
            available_consensus_tickers = [t for t in episode_consensus_tickers if t in execution_prices]
            
            logger.debug(f"Episode {i} ({episode.date}): {len(episode_consensus_tickers)} episode consensus tickers, "
                        f"{len(available_consensus_tickers)} available for trading")
            
            # Get the appropriate pretrained model for this episode
            decision_maker = model_manager.get_decision_maker_for_episode(i)
            if not decision_maker:
                raise ValueError(f"No pretrained model available for episode {i}. Check your pretrained models.")
            
            logger.debug(f"Using pretrained model for episode {i}")
            
            # Make LSTM trading decisions only for consensus tickers
            recent_episodes = episodes[max(0, i - self.sequence_length):i + 1]
            
            lstm_decisions = decision_maker.make_decisions(
                recent_episodes=recent_episodes,
                available_tickers=available_consensus_tickers,
                portfolio_state=portfolio_before,
                risk_limits=risk_limits  
            )
            
            # Extract bond ETF decisions from episode LLM decisions (fixed income agent allocation)
            bond_etf_decisions = {}
            for bond_etf in ["SHY", "TLT"]:
                if bond_etf in episode.llm_decisions and bond_etf in execution_prices:
                    bond_decision = episode.llm_decisions[bond_etf]
                    bond_etf_decisions[bond_etf] = {
                        'action': bond_decision.action,
                        'quantity': bond_decision.quantity,
                        'confidence': bond_decision.confidence,
                        'reasoning': bond_decision.reasoning
                    }
                    logger.debug(f"Bond ETF decision for {bond_etf}: {bond_decision.action} {bond_decision.quantity} shares")
            
            # Combine LSTM decisions with bond ETF decisions
            all_decisions = {**lstm_decisions, **bond_etf_decisions}
            
            # Execute trades using portfolio engine (execution_prices already defined above)
            executed_trades = {}
            
            # First, close positions for tickers not in current episode consensus
            # This prevents large legacy positions from persisting when they're no longer being analyzed
            for ticker, position in portfolio_engine.portfolio["positions"].items():
                if (position["long"] > 0 or position["short"] > 0) and ticker not in available_consensus_tickers:
                    if ticker in execution_prices:  # Can only trade if we have prices
                        logger.info(f"Closing legacy position in {ticker}: not in current episode consensus")
                        
                        # Close long positions
                        if position["long"] > 0:
                            executed_qty = portfolio_engine.execute_trade(
                                ticker, "sell", position["long"], execution_prices[ticker]
                            )
                            executed_trades[ticker] = executed_qty
                            logger.info(f"Closed long position: sold {executed_qty} shares of {ticker}")
                        
                        # Close short positions
                        if position["short"] > 0:
                            executed_qty = portfolio_engine.execute_trade(
                                ticker, "cover", position["short"], execution_prices[ticker]
                            )
                            executed_trades[ticker] = executed_qty
                            logger.info(f"Closed short position: covered {executed_qty} shares of {ticker}")
            
            # Then execute all decisions (LSTM + bond ETF decisions)
            for ticker, decision in all_decisions.items():
                if ticker in execution_prices:
                    # Handle position limits like the main backtester
                    action = decision['action']
                    quantity = decision['quantity']
                    
                    # Validate quantity against risk management limits (remaining_position_limit is in dollars)
                    # NOTE: During episode data collection, remaining_position_limit was effectively an absolute 
                    # position limit due to a bug in risk_manager.py that always returned 0 for current_position_value
                    # Bond ETFs don't have risk limits, so skip this validation for them
                    if ticker in risk_limits and action in ['buy', 'short'] and quantity > 0:
                        remaining_limit = risk_limits[ticker].get('remaining_position_limit', 0.0)  # Dollar limit
                        current_price = execution_prices[ticker]
                        
                        # Check current total position size in dollars
                        existing_position = portfolio_engine.portfolio["positions"][ticker]
                        current_long_value = existing_position.get("long", 0) * current_price
                        current_short_value = existing_position.get("short", 0) * current_price  
                        current_total_position_value = current_long_value + current_short_value  # Total exposure in dollars
                        
                        # Calculate how much more we can add without exceeding the limit
                        available_limit = remaining_limit - current_total_position_value
                        
                        if available_limit <= 0:
                            logger.info(f"Position limit reached for {ticker}: current position ${current_total_position_value:.0f}, "
                                      f"limit ${remaining_limit:.0f}. Skipping {action}.")
                            continue
                        
                        # Calculate max additional shares we can trade within the remaining limit
                        max_additional_shares = int(available_limit / current_price) if current_price > 0 else 0
                        
                        if quantity > max_additional_shares:
                            logger.info(f"Risk limit for {ticker}: reducing {action} from {quantity} to {max_additional_shares} shares "
                                      f"(${available_limit:.0f} remaining of ${remaining_limit:.0f} limit)")
                            quantity = max_additional_shares
                            
                        if quantity <= 0:
                            logger.debug(f"Skipping {ticker} {action}: would exceed dollar position limit")
                            continue
                    
                    if action in ['buy', 'short'] and quantity > 0:
                        # Check position limits (bond ETFs don't count toward position limits)
                        existing_position = portfolio_engine.portfolio["positions"][ticker]
                        has_existing_position = existing_position["long"] > 0 or existing_position["short"] > 0
                        
                        if not has_existing_position and ticker not in portfolio_engine.bond_etfs:
                            # Check if we should open new position (only for non-bond ETFs)
                            signal_strength = portfolio_engine.calculate_signal_strength(
                                episode.agent_signals, ticker
                            )
                            should_open, ticker_to_close = portfolio_engine.should_open_position(
                                ticker, signal_strength, episode.agent_signals
                            )
                            
                            if should_open:
                                # Close weakest position if needed
                                if ticker_to_close:
                                    close_position = portfolio_engine.portfolio["positions"][ticker_to_close]
                                    if close_position["long"] > 0:
                                        close_qty = portfolio_engine.execute_trade(
                                            ticker_to_close, "sell", close_position["long"], 
                                            execution_prices[ticker_to_close]
                                        )
                                        executed_trades[ticker_to_close] = close_qty
                                    elif close_position["short"] > 0:
                                        close_qty = portfolio_engine.execute_trade(
                                            ticker_to_close, "cover", close_position["short"],
                                            execution_prices[ticker_to_close]
                                        )
                                        executed_trades[ticker_to_close] = close_qty
                                
                                # Execute new trade
                                executed_qty = portfolio_engine.execute_trade(
                                    ticker, action, quantity, execution_prices[ticker]
                                )
                                executed_trades[ticker] = executed_qty
                            # else: position limit reached, don't trade
                        elif ticker in portfolio_engine.bond_etfs:
                            # Bond ETFs always execute (no position limits)
                            executed_qty = portfolio_engine.execute_trade(
                                ticker, action, quantity, execution_prices[ticker]
                            )
                            executed_trades[ticker] = executed_qty
                        else:
                            # Existing position, execute normally
                            executed_qty = portfolio_engine.execute_trade(
                                ticker, action, quantity, execution_prices[ticker]
                            )
                            executed_trades[ticker] = executed_qty
                    else:
                        # Sell/cover actions
                        executed_qty = portfolio_engine.execute_trade(
                            ticker, action, quantity, execution_prices[ticker]
                        )
                        executed_trades[ticker] = executed_qty
            
            # Calculate portfolio value after trading
            portfolio_after = portfolio_engine.get_portfolio_state_copy()
            # Value portfolio at current day's close prices
            portfolio_value_after = portfolio_engine.get_portfolio_value(evaluation_prices)
            
            # Calculate daily return
            daily_return = portfolio_engine.calculate_portfolio_return(
                portfolio_value_before, portfolio_value_after
            )
            
            # ---------------------------------------------------------------
            # Generate daily position table (similar to src/backtester.py)
            # ---------------------------------------------------------------
            date_rows = []
            
            # Use original episode analyst signals for display (same signals LSTM was trained on)
            analyst_signals = episode.agent_signals
            
            # Create table rows for tickers with positions or trading activity
            # Include: current consensus tickers + bond ETFs + any tickers that had trades executed
            all_relevant_tickers = set(available_consensus_tickers)
            # Always include bond ETFs if they have data
            for bond_etf in ["SHY", "TLT"]:
                if bond_etf in execution_prices and bond_etf in evaluation_prices:
                    all_relevant_tickers.add(bond_etf)
            
            # Add tickers with current positions
            for ticker, position in portfolio_engine.portfolio["positions"].items():
                if position["long"] > 0 or position["short"] > 0:
                    all_relevant_tickers.add(ticker)
            
            # Add tickers that had trades executed (including position closures)
            for ticker in executed_trades.keys():
                all_relevant_tickers.add(ticker)
            
            for ticker in all_relevant_tickers:
                # Skip tickers without current market data
                if ticker not in evaluation_prices or ticker not in execution_prices:
                    continue
                    
                position = portfolio_engine.portfolio["positions"][ticker]
                shares_owned = position["long"] - position["short"]  # net shares
                
                # Calculate position value
                long_val = position["long"] * evaluation_prices[ticker]
                short_val = position["short"] * evaluation_prices[ticker]  
                net_position_value = long_val - short_val
                
                # Calculate position return percentage
                position_return_pct = None
                if position["long"] > 0 and position["long_cost_basis"] > 0:
                    position_return_pct = ((evaluation_prices[ticker] - position["long_cost_basis"]) / position["long_cost_basis"]) * 100
                elif position["short"] > 0 and position["short_cost_basis"] > 0:
                    position_return_pct = ((position["short_cost_basis"] - evaluation_prices[ticker]) / position["short_cost_basis"]) * 100
                
                # Get action and executed quantity
                decision = all_decisions.get(ticker, {})
                quantity = executed_trades.get(ticker, 0)
                
                # Determine action - could be from LSTM decision, bond ETF decision, or legacy position closure
                if ticker in all_decisions:
                    action = all_decisions[ticker].get("action", "hold")
                elif quantity != 0:
                    # This was a legacy position closure (not in current episode consensus)
                    if quantity > 0:
                        action = "SELL"  # Closed long position
                    else:
                        action = "COVER"  # Closed short position
                    quantity = abs(quantity)  # Show positive quantity for display
                else:
                    action = "hold"
                
                # Count analyst signals for this ticker (from original episode data)
                ticker_signals = {}
                for agent_name, signals in analyst_signals.items():
                    if agent_name == "risk_management_agent":
                        continue  # Skip risk management signals for display
                    if ticker in signals:
                        ticker_signals[agent_name] = signals[ticker]
                
                bullish_count = len([s for s in ticker_signals.values() if hasattr(s, 'signal') and s.signal == "bullish"])
                bearish_count = len([s for s in ticker_signals.values() if hasattr(s, 'signal') and s.signal == "bearish"])  
                neutral_count = len([s for s in ticker_signals.values() if hasattr(s, 'signal') and s.signal == "neutral"])
                
                # Only show tickers with open positions or current trading activity
                if shares_owned != 0 or quantity != 0:
                    date_rows.append(
                        format_backtest_row(
                            date=episode.date,
                            ticker=ticker,
                            action=action,
                            quantity=quantity,
                            open_price=execution_prices[ticker],
                            close_price=evaluation_prices[ticker],
                            shares_owned=shares_owned,
                            position_value=net_position_value,
                            bullish_count=bullish_count,
                            bearish_count=bearish_count,
                            neutral_count=neutral_count,
                            position_return_pct=position_return_pct,
                        )
                    )
            
            # Calculate cumulative return for summary
            portfolio_return = ((portfolio_value_after) / self.initial_capital - 1) * 100
            
            # Add summary row for this day
            date_rows.append(
                format_backtest_row(
                    date=episode.date,
                    ticker="",
                    action="",
                    quantity=0,
                    open_price=0,
                    close_price=0,
                    shares_owned=0,
                    position_value=0,
                    bullish_count=0,
                    bearish_count=0,
                    neutral_count=0,
                    is_summary=True,
                    total_value=portfolio_value_after,
                    return_pct=portfolio_return,
                    cash_balance=portfolio_engine.portfolio["cash"],
                    total_position_value=portfolio_value_after - portfolio_engine.portfolio["cash"],
                    sharpe_ratio=None,  # Will be calculated at the end
                    sortino_ratio=None,
                    max_drawdown=None,
                ),
            )
            
            # Display the daily table
            print_backtest_results(date_rows)
            
            # Record results
            daily_result = {
                'date': episode.date,
                'episode_index': i,
                'portfolio_value_before': portfolio_value_before,
                'portfolio_value_after': portfolio_value_after,
                'daily_return': daily_return,
                'lstm_decisions': lstm_decisions,
                'bond_etf_decisions': bond_etf_decisions,
                'all_decisions': all_decisions,
                'executed_trades': executed_trades,
                'llm_return': episode.portfolio_return,  # For comparison
                'table_rows': date_rows  # Include table data in results
            }
            
            daily_results.append(daily_result)
            current_portfolio_value = portfolio_value_after
            
            # Track portfolio values for plotting
            self.portfolio_values.append({
                'Date': episode.date,
                'Portfolio Value': portfolio_value_after,
                'Daily Return': daily_return
            })
        
        # Calculate final performance metrics
        returns_series = pd.Series([r['daily_return'] for r in daily_results])
        llm_returns = pd.Series([r['llm_return'] for r in daily_results])
        
        performance_metrics = {
            'total_episodes': len(daily_results),
            'initial_capital': self.initial_capital,
            'final_portfolio_value': current_portfolio_value,
            'total_return': (current_portfolio_value - self.initial_capital) / self.initial_capital,
            'lstm_sharpe_ratio': self._calculate_sharpe_ratio(returns_series),
            'lstm_volatility': returns_series.std() * np.sqrt(252),
            'llm_total_return': (1 + llm_returns).prod() - 1,
            'llm_sharpe_ratio': self._calculate_sharpe_ratio(llm_returns),
            'llm_volatility': llm_returns.std() * np.sqrt(252),
            'pretrained_models_used': True
        }
        
        self.results = daily_results
        
        logger.info("Walk-forward backtest completed!")
        logger.info(f"LSTM Total Return: {performance_metrics['total_return']:.2%}")
        logger.info(f"LSTM Sharpe Ratio: {performance_metrics['lstm_sharpe_ratio']:.3f}")
        logger.info(f"LLM Total Return: {performance_metrics['llm_total_return']:.2%}")
        logger.info(f"LLM Sharpe Ratio: {performance_metrics['llm_sharpe_ratio']:.3f}")
        
        return {
            'daily_results': daily_results,
            'performance_metrics': performance_metrics,
            'model_performance': self.model_performance,
            'portfolio_values': self.portfolio_values
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from returns series."""
        if returns.std() == 0:
            return 0.0
        
        annual_return = returns.mean() * 252  # Assume daily returns
        annual_volatility = returns.std() * np.sqrt(252)
        
        return (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0
    
    def export_results_to_csv(self, filename: str = None) -> str:
        """Export LSTM backtesting results to CSV for benchmarking."""
        if not self.results:
            raise ValueError("No results to export. Run backtest first.")
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                'date': result['date'],
                'return': result['daily_return']
            }
            for result in self.results
        ])
        
        # Create filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lstm_walkforward_returns_{timestamp}.csv"
        
        # Ensure benchmarking directory exists
        benchmarking_dir = Path("benchmarking/data")
        benchmarking_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        csv_path = benchmarking_dir / filename
        df.to_csv(csv_path, index=False)
        
        logger.info(f"LSTM walk-forward results exported to: {csv_path}")
        return str(csv_path)


def main():
    """Main function to run LSTM walk-forward backtesting."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize backtester
    backtester = LSTMWalkForwardBacktester(
        initial_capital=100000.0,
        initial_training_episodes=250,
        retraining_frequency=30,  # Match pretrained model frequency (not used but kept for compatibility)
        max_positions=30,
        transaction_cost=0.0005,
        sequence_length=20
    )
    
    # Load episodes
    episodes = backtester.load_all_episodes()
    
    if len(episodes) < 300:
        logger.error(f"Insufficient episodes: {len(episodes)}")
        return
    
    # Run walk-forward backtest
    results = backtester.run_walk_forward_backtest(episodes)
    
    # Export results
    csv_path = backtester.export_results_to_csv()
    
    # Print summary
    metrics = results['performance_metrics']
    print(f"\nLSTM Walk-Forward Backtest Results:")
    print(f"  Episodes processed: {metrics['total_episodes']}")
    print(f"  Final portfolio value: ${metrics['final_portfolio_value']:,.2f}")
    print(f"  LSTM Total Return: {metrics['total_return']:.2%}")
    print(f"  LSTM Sharpe Ratio: {metrics['lstm_sharpe_ratio']:.3f}")
    print(f"  LSTM Volatility: {metrics['lstm_volatility']:.2%}")
    print(f"  Results saved to: {csv_path}")


if __name__ == "__main__":
    main()