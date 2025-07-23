"""
Episode collector for gathering training data during backtesting runs.

This module handles the collection and storage of training episodes
that will be used to train the LSTM reinforcement learning model.
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

from .episode import TrainingEpisode, AgentSignal, LLMDecision, PortfolioState
from utils.logger import logger


class EpisodeCollector:
    """
    Collects and manages training episodes during backtesting runs.
    
    This class is responsible for:
    1. Capturing data from the hedge fund workflow
    2. Converting it to TrainingEpisode objects
    3. Storing episodes to disk
    4. Loading episodes for training
    """
    
    def __init__(self, output_dir: str = "training_data", format: str = "json"):
        """
        Initialize the episode collector.
        
        Args:
            output_dir: Directory to store training episodes
            format: Storage format ('json' or 'pickle')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self.episodes: List[TrainingEpisode] = []
        
        # Create subdirectories for organized storage
        (self.output_dir / "episodes").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        logger.info(f"Episode collector initialized with output directory: {self.output_dir}")
    
    def collect_episode(
        self,
        date: str,
        agent_signals: Dict[str, Dict[str, Dict[str, Any]]],
        llm_decisions: Dict[str, Dict[str, Any]],
        portfolio_before: Dict[str, Any],
        portfolio_after: Dict[str, Any],
        portfolio_return: float,
        market_data: Optional[Dict[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrainingEpisode:
        """
        Collect a single training episode.
        
        Args:
            date: Trading date in YYYY-MM-DD format
            agent_signals: Raw agent signals from the workflow
            llm_decisions: Raw LLM decisions from equity agent
            portfolio_before: Portfolio state before trading
            portfolio_after: Portfolio state after trading
            portfolio_return: Daily portfolio return
            market_data: Additional market context data
            metadata: Metadata about the run (model, analysts, etc.)
            
        Returns:
            TrainingEpisode object
        """
        logger.debug(f"Collecting episode for date: {date}")
        
        # Convert agent signals to AgentSignal objects
        processed_signals = {}
        
        # Define agent types based on their signal structure
        portfolio_level_agents = {"macro_agent", "forward_looking_agent", "fixed_income_agent"}
        ticker_based_signal_agents = {"sentiment_agent", "technical_analyst_agent", "fundamentals_agent", 
                                     "valuation_agent", "bill_ackman_agent", "warren_buffett_agent"}
        ticker_based_other_agents = {"risk_management_agent"}  # Has ticker keys but different structure
        
        for agent_name, agent_data in agent_signals.items():
            processed_signals[agent_name] = {}
            
            if agent_name in portfolio_level_agents:
                # Portfolio-level agents: store raw data with special key
                processed_signals[agent_name]["_portfolio_data"] = agent_data
            elif agent_name in ticker_based_signal_agents:
                # Standard ticker-based agents with signal/confidence structure
                if isinstance(agent_data, dict):
                    for ticker, signal_data in agent_data.items():
                        if isinstance(signal_data, dict) and 'signal' in signal_data:
                            processed_signals[agent_name][ticker] = AgentSignal(
                                signal=signal_data.get('signal', 'neutral'),
                                confidence=signal_data.get('confidence', 0.0)
                            )
                        else:
                            logger.warning(f"Unexpected signal format for {agent_name}[{ticker}]: {signal_data}")
            elif agent_name in ticker_based_other_agents:
                # Ticker-based agents with different structure (like risk_management_agent)
                if isinstance(agent_data, dict):
                    for ticker, ticker_data in agent_data.items():
                        # Store the raw ticker data - these agents don't have signal/confidence
                        processed_signals[agent_name][ticker] = ticker_data
            else:
                # Unknown agent type - log warning and store raw data
                logger.warning(f"Unknown agent type: {agent_name}, storing raw data")
                processed_signals[agent_name]["_raw_data"] = agent_data
        
        # Convert LLM decisions to LLMDecision objects
        processed_decisions = {}
        for ticker, decision_data in llm_decisions.items():
            processed_decisions[ticker] = LLMDecision(
                action=decision_data.get('action', 'hold'),
                quantity=decision_data.get('quantity', 0),
                confidence=decision_data.get('confidence', 0.0),
                reasoning=decision_data.get('reasoning', '')
            )
        
        # Create portfolio state objects - handle both dict and object inputs
        if hasattr(portfolio_before, 'to_dict'):
            portfolio_before_dict = portfolio_before.to_dict()
        else:
            portfolio_before_dict = portfolio_before
        
        if hasattr(portfolio_after, 'to_dict'):
            portfolio_after_dict = portfolio_after.to_dict()
        else:
            portfolio_after_dict = portfolio_after
        
        # Convert portfolio structure to match PortfolioState expected format
        def normalize_portfolio(portfolio_dict):
            return {
                'cash': portfolio_dict.get('cash', 0.0),
                'positions': portfolio_dict.get('positions', {}),
                'realized_gains': portfolio_dict.get('realized_gains', {}),
                'margin_requirement': portfolio_dict.get('margin_used', portfolio_dict.get('margin_requirement', 0.0))
            }
        
        portfolio_before_obj = PortfolioState.from_dict(normalize_portfolio(portfolio_before_dict))
        portfolio_after_obj = PortfolioState.from_dict(normalize_portfolio(portfolio_after_dict))
        
        # Create the training episode
        episode = TrainingEpisode(
            date=date,
            agent_signals=processed_signals,
            llm_decisions=processed_decisions,
            portfolio_before=portfolio_before_obj,
            portfolio_after=portfolio_after_obj,
            portfolio_return=portfolio_return,
            market_data=market_data or {},
            metadata=metadata or {}
        )
        
        self.episodes.append(episode)
        logger.debug(f"Episode collected for {date}. Total episodes: {len(self.episodes)}")
        
        return episode
    
    def save_episode(self, episode: TrainingEpisode) -> Path:
        """
        Save a single episode to disk.
        
        Args:
            episode: TrainingEpisode to save
            
        Returns:
            Path to saved file
        """
        filename = f"episode_{episode.date}.{self.format}"
        filepath = self.output_dir / "episodes" / filename
        
        if self.format == "json":
            episode.save_json(filepath)
        elif self.format == "pickle":
            episode.save_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {self.format}")
        
        logger.debug(f"Episode saved to: {filepath}")
        return filepath
    
    def save_all_episodes(self) -> List[Path]:
        """
        Save all collected episodes to disk.
        
        Returns:
            List of paths to saved files
        """
        saved_files = []
        for episode in self.episodes:
            filepath = self.save_episode(episode)
            saved_files.append(filepath)
        
        logger.info(f"Saved {len(saved_files)} episodes to {self.output_dir}")
        return saved_files
    
    def load_episodes(self, date_range: Optional[tuple] = None) -> List[TrainingEpisode]:
        """
        Load episodes from disk.
        
        Args:
            date_range: Optional tuple of (start_date, end_date) to filter episodes
            
        Returns:
            List of loaded TrainingEpisode objects
        """
        episodes_dir = self.output_dir / "episodes"
        if not episodes_dir.exists():
            logger.warning(f"Episodes directory not found: {episodes_dir}")
            return []
        
        episodes = []
        pattern = f"episode_*.{self.format}"
        
        for filepath in episodes_dir.glob(pattern):
            try:
                if self.format == "json":
                    episode = TrainingEpisode.load_json(filepath)
                elif self.format == "pickle":
                    episode = TrainingEpisode.load_pickle(filepath)
                else:
                    continue
                
                # Filter by date range if specified
                if date_range:
                    start_date, end_date = date_range
                    if not (start_date <= episode.date <= end_date):
                        continue
                
                episodes.append(episode)
                
            except Exception as e:
                logger.error(f"Error loading episode from {filepath}: {e}")
                continue
        
        # Sort episodes by date
        episodes.sort(key=lambda x: x.date)
        logger.info(f"Loaded {len(episodes)} episodes from {episodes_dir}")
        
        return episodes
    
    def create_summary_report(self) -> Dict[str, Any]:
        """
        Create a summary report of collected episodes.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.episodes:
            return {"total_episodes": 0, "message": "No episodes collected"}
        
        # Basic statistics
        total_episodes = len(self.episodes)
        date_range = (self.episodes[0].date, self.episodes[-1].date)
        
        # Performance statistics
        returns = [ep.portfolio_return for ep in self.episodes]
        avg_return = sum(returns) / len(returns)
        
        # Signal statistics
        signal_counts = {}
        for episode in self.episodes:
            for agent_name, ticker_signals in episode.agent_signals.items():
                if agent_name not in signal_counts:
                    signal_counts[agent_name] = {"bullish": 0, "bearish": 0, "neutral": 0}
                
                for ticker, signal in ticker_signals.items():
                    # Skip special keys and non-AgentSignal objects
                    if ticker.startswith("_") or not hasattr(signal, 'signal'):
                        continue
                    signal_counts[agent_name][signal.signal] += 1
        
        # Decision statistics
        decision_counts = {"buy": 0, "sell": 0, "hold": 0, "short": 0, "cover": 0}
        for episode in self.episodes:
            for ticker, decision in episode.llm_decisions.items():
                decision_counts[decision.action] += 1
        
        summary = {
            "total_episodes": total_episodes,
            "date_range": date_range,
            "avg_daily_return": avg_return,
            "signal_counts": signal_counts,
            "decision_counts": decision_counts,
            "storage_format": self.format,
            "output_directory": str(self.output_dir)
        }
        
        return summary
    
    def save_summary_report(self) -> Path:
        """
        Save summary report to disk.
        
        Returns:
            Path to saved report
        """
        summary = self.create_summary_report()
        report_path = self.output_dir / "metadata" / "summary_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to: {report_path}")
        return report_path
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert episodes to a pandas DataFrame for analysis.
        
        Returns:
            DataFrame with episode data
        """
        if not self.episodes:
            return pd.DataFrame()
        
        data = []
        for episode in self.episodes:
            # Basic episode data
            row = {
                'date': episode.date,
                'portfolio_return': episode.portfolio_return,
                'cash_before': episode.portfolio_before.cash,
                'cash_after': episode.portfolio_after.cash,
            }
            
            # Add metadata
            row.update(episode.metadata)
            
            # Add aggregated signal statistics
            for ticker in episode.llm_decisions.keys():
                consensus = episode.get_signal_consensus(ticker)
                row[f'{ticker}_consensus'] = consensus['consensus']
                row[f'{ticker}_strength'] = consensus['strength']
                row[f'{ticker}_agreement'] = consensus['agreement']
                row[f'{ticker}_avg_confidence'] = consensus['avg_confidence']
                
                # Add LLM decision
                decision = episode.llm_decisions[ticker]
                row[f'{ticker}_action'] = decision.action
                row[f'{ticker}_quantity'] = decision.quantity
                row[f'{ticker}_decision_confidence'] = decision.confidence
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def clear_episodes(self):
        """Clear all collected episodes from memory."""
        self.episodes.clear()
        logger.info("Cleared all episodes from memory")