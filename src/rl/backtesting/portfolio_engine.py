"""
Portfolio Execution Engine for LSTM Trading System.

This module contains the PortfolioEngine class extracted from the main backtester
to provide identical portfolio execution logic for LSTM research.
"""

import logging
from typing import Dict, Any, Set, Tuple, Optional

logger = logging.getLogger(__name__)


class PortfolioEngine:
    """
    Portfolio execution engine that handles trading operations identically to the main backtester.
    
    This class is extracted from the main Backtester class to provide standalone portfolio
    execution capabilities for LSTM research without requiring the full backtesting infrastructure.
    """
    
    def __init__(
        self,
        initial_capital: float,
        all_tickers: list[str],
        max_positions: int = 30,
        transaction_cost: float = 0.0005,
        margin_ratio: float = 0.0,
        bond_etfs: list[str] = None
    ):
        """
        Initialize the portfolio engine.
        
        Args:
            initial_capital: Starting portfolio cash
            all_tickers: List of all tickers (stocks + bond ETFs)
            max_positions: Maximum number of active positions allowed
            transaction_cost: Transaction cost as percentage of trade value (e.g. 0.0005 = 5 basis points)
            margin_ratio: Margin requirement for short positions (e.g. 0.5 = 50%)
            bond_etfs: List of bond ETF tickers (don't count toward position limits)
        """
        self.initial_capital = initial_capital
        self.all_tickers = all_tickers
        self.max_positions = max_positions
        self.transaction_cost = transaction_cost
        self.margin_ratio = margin_ratio
        self.bond_etfs = bond_etfs or ["SHY", "TLT"]
        
        # Initialize portfolio with support for long/short positions
        self.portfolio = {
            "cash": initial_capital,
            "margin_used": 0.0,  # total margin usage across all short positions
            "positions": {
                ticker: {
                    "long": 0,               # Number of shares held long
                    "short": 0,              # Number of shares held short
                    "long_cost_basis": 0.0,  # Average cost basis per share (long)
                    "short_cost_basis": 0.0, # Average cost basis per share (short)
                    "short_margin_used": 0.0 # Dollars of margin used for this ticker's short
                } for ticker in all_tickers
            },
            "realized_gains": {
                ticker: {
                    "long": 0.0,   # Realized gains from long positions
                    "short": 0.0,  # Realized gains from short positions
                } for ticker in all_tickers
            }
        }
        
        logger.info(f"PortfolioEngine initialized with ${initial_capital:.2f}, max_positions={max_positions}")
    
    def execute_trade(self, ticker: str, action: str, quantity: float, execution_price: float) -> int:
        """
        Execute trades with support for both long and short positions.
        
        Args:
            ticker: Stock ticker symbol
            action: Trading action ('buy', 'sell', 'short', 'cover', 'hold')
            quantity: Number of shares to trade
            execution_price: Price per share for execution
            
        Returns:
            Number of shares actually executed (may be less than requested due to constraints)
        """
        if quantity <= 0 or action == "hold":
            return 0

        quantity = int(quantity)  # force integer shares
        position = self.portfolio["positions"][ticker]

        if action == "buy":
            cost = quantity * execution_price
            transaction_fee = cost * self.transaction_cost
            total_cost = cost + transaction_fee
            
            if total_cost <= self.portfolio["cash"]:
                # Weighted average cost basis for the new total
                old_shares = position["long"]
                old_cost_basis = position["long_cost_basis"]
                new_shares = quantity
                total_shares = old_shares + new_shares

                if total_shares > 0:
                    total_old_cost = old_cost_basis * old_shares
                    total_new_cost = cost
                    position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                position["long"] += quantity
                self.portfolio["cash"] -= total_cost
                return quantity
            else:
                # Calculate maximum affordable quantity including transaction costs
                max_quantity = int(self.portfolio["cash"] / (execution_price * (1 + self.transaction_cost)))
                if max_quantity > 0:
                    cost = max_quantity * execution_price
                    transaction_fee = cost * self.transaction_cost
                    total_cost = cost + transaction_fee
                    
                    old_shares = position["long"]
                    old_cost_basis = position["long_cost_basis"]
                    total_shares = old_shares + max_quantity

                    if total_shares > 0:
                        total_old_cost = old_cost_basis * old_shares
                        total_new_cost = cost
                        position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                    position["long"] += max_quantity
                    self.portfolio["cash"] -= total_cost
                    return max_quantity
                return 0

        elif action == "sell":
            # You can only sell as many as you own
            quantity = min(quantity, position["long"])
            if quantity > 0:
                proceeds = quantity * execution_price
                transaction_fee = proceeds * self.transaction_cost
                net_proceeds = proceeds - transaction_fee
                
                # Realized gain/loss using average cost basis
                avg_cost_per_share = position["long_cost_basis"] if position["long"] > 0 else 0
                realized_gain = (execution_price - avg_cost_per_share) * quantity - transaction_fee
                self.portfolio["realized_gains"][ticker]["long"] += realized_gain

                position["long"] -= quantity
                self.portfolio["cash"] += net_proceeds

                if position["long"] == 0:
                    position["long_cost_basis"] = 0.0

                return quantity

        elif action == "short":
            """
            Typical short sale flow:
              1) Receive proceeds = execution_price * quantity
              2) Pay transaction fee = proceeds * transaction_cost
              3) Post margin_required = proceeds * margin_ratio
              4) Net effect on cash = +proceeds - transaction_fee - margin_required
            """
            proceeds = execution_price * quantity
            transaction_fee = proceeds * self.transaction_cost
            margin_required = proceeds * self.margin_ratio
            net_cash_needed = margin_required + transaction_fee
            
            if net_cash_needed <= self.portfolio["cash"]:
                # Weighted average short cost basis
                old_short_shares = position["short"]
                old_cost_basis = position["short_cost_basis"]
                new_shares = quantity
                total_shares = old_short_shares + new_shares

                if total_shares > 0:
                    total_old_cost = old_cost_basis * old_short_shares
                    total_new_cost = execution_price * new_shares
                    position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                position["short"] += quantity

                # Update margin usage
                position["short_margin_used"] += margin_required
                self.portfolio["margin_used"] += margin_required

                # Increase cash by proceeds, then subtract transaction fee and required margin
                self.portfolio["cash"] += proceeds
                self.portfolio["cash"] -= transaction_fee
                self.portfolio["cash"] -= margin_required
                return quantity
            else:
                # Calculate maximum shortable quantity including transaction costs
                if self.margin_ratio > 0:
                    max_quantity = int(self.portfolio["cash"] / (execution_price * (self.margin_ratio + self.transaction_cost)))
                else:
                    max_quantity = int(self.portfolio["cash"] / (execution_price * self.transaction_cost))

                if max_quantity > 0:
                    proceeds = execution_price * max_quantity
                    transaction_fee = proceeds * self.transaction_cost
                    margin_required = proceeds * self.margin_ratio

                    old_short_shares = position["short"]
                    old_cost_basis = position["short_cost_basis"]
                    total_shares = old_short_shares + max_quantity

                    if total_shares > 0:
                        total_old_cost = old_cost_basis * old_short_shares
                        total_new_cost = execution_price * max_quantity
                        position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                    position["short"] += max_quantity
                    position["short_margin_used"] += margin_required
                    self.portfolio["margin_used"] += margin_required

                    self.portfolio["cash"] += proceeds
                    self.portfolio["cash"] -= transaction_fee
                    self.portfolio["cash"] -= margin_required
                    return max_quantity
                return 0

        elif action == "cover":
            """
            When covering shares:
              1) Pay cover cost = execution_price * quantity
              2) Pay transaction fee = cover_cost * transaction_cost
              3) Release a proportional share of the margin
              4) Net effect on cash = -cover_cost - transaction_fee + released_margin
            """
            quantity = min(quantity, position["short"])
            if quantity > 0:
                cover_cost = quantity * execution_price
                transaction_fee = cover_cost * self.transaction_cost
                total_cost = cover_cost + transaction_fee
                
                avg_short_price = position["short_cost_basis"] if position["short"] > 0 else 0
                realized_gain = (avg_short_price - execution_price) * quantity - transaction_fee

                if position["short"] > 0:
                    portion = quantity / position["short"]
                else:
                    portion = 1.0

                margin_to_release = portion * position["short_margin_used"]

                position["short"] -= quantity
                position["short_margin_used"] -= margin_to_release
                self.portfolio["margin_used"] -= margin_to_release

                # Pay the cost to cover plus transaction fee, but get back the released margin
                self.portfolio["cash"] += margin_to_release
                self.portfolio["cash"] -= total_cost

                self.portfolio["realized_gains"][ticker]["short"] += realized_gain

                if position["short"] == 0:
                    position["short_cost_basis"] = 0.0
                    position["short_margin_used"] = 0.0

                return quantity

        return 0

    def get_open_positions(self, exclude_bond_etfs: bool = False) -> Set[str]:
        """Get all tickers with open positions (long or short)."""
        open_positions = set()
        for ticker, position in self.portfolio["positions"].items():
            if position["long"] != 0 or position["short"] != 0:
                # Exclude bond ETFs from position count if requested
                if exclude_bond_etfs and ticker in self.bond_etfs:
                    continue
                open_positions.add(ticker)
        return open_positions

    def get_position_count(self, exclude_bond_etfs: bool = False) -> int:
        """Get the current number of active positions."""
        return len(self.get_open_positions(exclude_bond_etfs=exclude_bond_etfs))
        
    def get_stock_position_count(self) -> int:
        """Get the current number of stock positions (excluding bond ETFs)."""
        return self.get_position_count(exclude_bond_etfs=True)

    def calculate_signal_strength(self, analyst_signals: Dict[str, Dict[str, Any]], ticker: str) -> float:
        """Calculate signal strength for a ticker based on analyst signals."""
        signals = []
        for agent_name, agent_signals in analyst_signals.items():
            if ticker in agent_signals:
                agent_signal = agent_signals[ticker]
                # Handle both AgentSignal objects and dict formats
                if hasattr(agent_signal, 'signal'):
                    signal = agent_signal.signal
                else:
                    signal = agent_signal.get("signal", "neutral")
                    
                if signal == "bullish":
                    signals.append(1.0)
                elif signal == "bearish":
                    signals.append(-1.0)
                else:
                    signals.append(0.0)
        
        if not signals:
            return 0.0
        
        # Calculate signal strength as consensus strength
        avg_signal = sum(signals) / len(signals)
        consensus_strength = abs(avg_signal)
        
        # Boost strength based on number of agreeing signals
        agreeing_count = sum(1 for s in signals if (s > 0 and avg_signal > 0) or (s < 0 and avg_signal < 0))
        agreement_factor = agreeing_count / len(signals) if signals else 0
        
        # Final strength combines consensus and agreement
        return consensus_strength * agreement_factor

    def find_weakest_position(self, analyst_signals: Dict[str, Dict[str, Dict]]) -> Tuple[Optional[str], float]:
        """Find the weakest open stock position based on current signal strength (excludes bond ETFs)."""
        open_positions = self.get_open_positions(exclude_bond_etfs=True)
        
        if not open_positions:
            return None, 0.0
        
        weakest_ticker = None
        weakest_strength = float('inf')
        
        # Create a list of (ticker, strength) pairs for debugging
        position_strengths = []
        
        for ticker in open_positions:
            strength = self.calculate_signal_strength(analyst_signals, ticker)
            position_strengths.append((ticker, strength))
            if strength < weakest_strength:
                weakest_strength = strength
                weakest_ticker = ticker
        
        # Log position strengths for debugging
        strength_info = ", ".join([f"{t}:{s:.3f}" for t, s in sorted(position_strengths, key=lambda x: x[1])])
        logger.debug(f"Position strengths for replacement consideration: {strength_info}")
        
        return weakest_ticker, weakest_strength

    def should_open_position(self, ticker: str, signal_strength: float, analyst_signals: Dict[str, Dict[str, Dict]]) -> Tuple[bool, Optional[str]]:
        """Determine if a new position should be opened given current constraints."""
        # For bond ETFs, always allow (they don't count toward position limit)
        if ticker in self.bond_etfs:
            return True, None
            
        # Only count stock positions toward the limit
        current_stock_positions = self.get_stock_position_count()
        
        # If under position limit, open normally
        if current_stock_positions < self.max_positions:
            return True, None
        
        # At position limit - need to compare with existing stock positions
        weakest_ticker, weakest_strength = self.find_weakest_position(analyst_signals)
        
        if weakest_ticker is None:
            logger.debug(f"No weakest position found for replacement - no open positions to replace")
            return False, None
        
        # If new signal is stronger than weakest position, replace it
        if signal_strength > weakest_strength:
            logger.debug(f"Replacing {weakest_ticker} (strength={weakest_strength:.3f}) with {ticker} (strength={signal_strength:.3f})")
            return True, weakest_ticker
        else:
            logger.debug(f"Not opening {ticker} (strength={signal_strength:.3f}) - weaker than {weakest_ticker} (strength={weakest_strength:.3f})")
            return False, None

    def get_portfolio_value(self, evaluation_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value using current market prices."""
        total_value = self.portfolio["cash"]
        
        for ticker, position in self.portfolio["positions"].items():
            if ticker in evaluation_prices:
                current_price = evaluation_prices[ticker]
                
                # Add value of long positions
                if position["long"] > 0:
                    total_value += position["long"] * current_price
                
                # Subtract value of short positions (liability)
                if position["short"] > 0:
                    total_value -= position["short"] * current_price
        
        return total_value
    
    def calculate_portfolio_return(self, previous_value: float, current_value: float) -> float:
        """Calculate portfolio return between two values."""
        if previous_value == 0:
            return 0.0
        return (current_value - previous_value) / previous_value
    
    def get_portfolio_state_copy(self) -> Dict[str, Any]:
        """Get a deep copy of the current portfolio state for episode recording."""
        import copy
        return copy.deepcopy(self.portfolio)
    
    def reset_portfolio(self):
        """Reset portfolio to initial state (useful for walk-forward validation)."""
        self.__init__(
            initial_capital=self.initial_capital,
            all_tickers=self.all_tickers,
            max_positions=self.max_positions,
            transaction_cost=self.transaction_cost,
            margin_ratio=self.margin_ratio,
            bond_etfs=self.bond_etfs
        )
        logger.info("Portfolio reset to initial state")