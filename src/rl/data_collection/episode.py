"""
Training episode data models for RL training.

This module defines the data structures used to capture training episodes
during hedge fund backtesting runs.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import pickle
from pathlib import Path


@dataclass
class AgentSignal:
    """Represents a signal from an individual agent."""
    signal: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal': self.signal,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentSignal':
        return cls(
            signal=data['signal'],
            confidence=data['confidence']
        )


@dataclass
class LLMDecision:
    """Represents a decision made by the LLM (equity agent)."""
    action: str  # 'buy', 'sell', 'short', 'cover', 'hold'
    quantity: int
    confidence: float  # 0.0 to 1.0
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'quantity': self.quantity,
            'confidence': self.confidence,
            'reasoning': self.reasoning
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMDecision':
        return cls(
            action=data['action'],
            quantity=data['quantity'],
            confidence=data['confidence'],
            reasoning=data['reasoning']
        )


@dataclass
class PortfolioState:
    """Represents the state of the portfolio."""
    cash: float
    positions: Dict[str, Dict[str, Any]]  # ticker -> {'long': int, 'short': int, 'long_cost_basis': float, 'short_cost_basis': float}
    realized_gains: Dict[str, Dict[str, float]]  # ticker -> {'long': float, 'short': float}
    margin_requirement: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cash': self.cash,
            'positions': self.positions,
            'realized_gains': self.realized_gains,
            'margin_requirement': self.margin_requirement
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioState':
        return cls(
            cash=data['cash'],
            positions=data['positions'],
            realized_gains=data['realized_gains'],
            margin_requirement=data.get('margin_requirement', 0.0)
        )


@dataclass
class TrainingEpisode:
    """
    Represents a complete training episode for a single trading day.
    
    This captures all the information needed to train the LSTM model including
    agent signals, LLM decisions, portfolio states, and performance metrics.
    """
    date: str  # Trading date in YYYY-MM-DD format
    agent_signals: Dict[str, Dict[str, AgentSignal]]  # agent_name -> ticker -> AgentSignal
    llm_decisions: Dict[str, LLMDecision]  # ticker -> LLMDecision
    portfolio_before: PortfolioState
    portfolio_after: PortfolioState
    portfolio_return: float  # Daily portfolio return
    market_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Additional market context
    metadata: Dict[str, Any] = field(default_factory=dict)  # Model used, analysts selected, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary for serialization."""
        return {
            'date': self.date,
            'agent_signals': {
                agent_name: {
                    ticker: signal.to_dict() if hasattr(signal, 'to_dict') else signal
                    for ticker, signal in ticker_signals.items()
                }
                for agent_name, ticker_signals in self.agent_signals.items()
            },
            'llm_decisions': {
                ticker: decision.to_dict() if hasattr(decision, 'to_dict') else decision
                for ticker, decision in self.llm_decisions.items()
            },
            'portfolio_before': self.portfolio_before.to_dict(),
            'portfolio_after': self.portfolio_after.to_dict(),
            'portfolio_return': self.portfolio_return,
            'market_data': self.market_data,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingEpisode':
        """Create episode from dictionary."""
        
        # Parse agent signals, handling different data structures
        agent_signals = {}
        for agent_name, agent_data in data['agent_signals'].items():
            agent_signals[agent_name] = {}
            
            for ticker, signal_data in agent_data.items():
                # Skip non-ticker entries (like _portfolio_data)
                if ticker.startswith('_'):
                    continue
                    
                # Only process if it looks like a signal (has 'signal' and 'confidence' keys)
                if isinstance(signal_data, dict) and 'signal' in signal_data and 'confidence' in signal_data:
                    agent_signals[agent_name][ticker] = AgentSignal.from_dict(signal_data)
        
        return cls(
            date=data['date'],
            agent_signals=agent_signals,
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
    
    def save_json(self, filepath: Path) -> None:
        """Save episode to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def save_pickle(self, filepath: Path) -> None:
        """Save episode to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_json(cls, filepath: Path) -> 'TrainingEpisode':
        """Load episode from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def load_pickle(cls, filepath: Path) -> 'TrainingEpisode':
        """Load episode from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_signal_consensus(self, ticker: str) -> Dict[str, Any]:
        """Get signal consensus statistics for a specific ticker."""
        signals = []
        confidences = []
        
        for agent_name, ticker_signals in self.agent_signals.items():
            if ticker in ticker_signals:
                signal = ticker_signals[ticker]
                # Only process AgentSignal objects
                if hasattr(signal, 'signal') and signal.signal != 'neutral':
                    signals.append(signal.signal)
                    confidences.append(signal.confidence)
        
        if not signals:
            return {'consensus': 'neutral', 'strength': 0.0, 'agreement': 0.0}
        
        bullish_count = signals.count('bullish')
        bearish_count = signals.count('bearish')
        total_signals = len(signals)
        
        if bullish_count > bearish_count:
            consensus = 'bullish'
            strength = bullish_count / total_signals
        elif bearish_count > bullish_count:
            consensus = 'bearish'
            strength = bearish_count / total_signals
        else:
            consensus = 'neutral'
            strength = 0.0
        
        agreement = max(bullish_count, bearish_count) / total_signals if total_signals > 0 else 0.0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'consensus': consensus,
            'strength': strength,
            'agreement': agreement,
            'avg_confidence': avg_confidence,
            'total_signals': total_signals
        }
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio metrics for this episode."""
        total_value_before = self.portfolio_before.cash
        total_value_after = self.portfolio_after.cash
        
        # Add position values (simplified - would need current prices in practice)
        # For now, we'll use the return directly
        
        return {
            'daily_return': self.portfolio_return,
            'cash_change': self.portfolio_after.cash - self.portfolio_before.cash,
            'total_value_before': total_value_before,
            'total_value_after': total_value_after
        }