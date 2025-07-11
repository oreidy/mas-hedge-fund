import matplotlib.pyplot as plt


import sys

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import questionary

import matplotlib.pyplot as plt
import pandas as pd
from colorama import Fore, Style, init
import numpy as np
import itertools

from llm.models import LLM_ORDER, get_model_info
from utils.analysts import ANALYST_ORDER
from main import run_hedge_fund
from tools.api import (
    get_company_news,
    get_price_data,
    get_prices,
    get_financial_metrics,
    get_insider_trades,
    get_data_for_tickers
)
from utils.market_calendar import is_trading_day, get_next_trading_day, get_previous_trading_day
from utils.display import print_backtest_results, format_backtest_row
from typing_extensions import Callable
import time
from utils.logger import logger
from utils.tickers import get_sp500_tickers

init(autoreset=True)


class Backtester:
    def __init__(
        self,
        agent: Callable,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        model_name: str = "gpt-4o",
        model_provider: str = "OpenAI",
        selected_analysts: list[str] = [],
        initial_margin_requirement: float = 0.0,
        debug_mode: bool = False,
        verbose_data: bool = False,
        screen_mode: bool = False,
    ):
        """
        :param agent: The trading agent (Callable).
        :param tickers: List of tickers to backtest.
        :param start_date: Start date string (YYYY-MM-DD).
        :param end_date: End date string (YYYY-MM-DD).
        :param initial_capital: Starting portfolio cash.
        :param model_name: Which LLM model name to use (gpt-4, etc).
        :param model_provider: Which LLM provider (OpenAI, etc).
        :param selected_analysts: List of analyst names or IDs to incorporate.
        :param initial_margin_requirement: The margin ratio (e.g. 0.5 = 50%).
        :param debug_mode: Whether to enable detailed debugging output.
        :param verbose_data: Whether to show detailed data output.
        :param screen_mode: Whether running in screening mode (vs specific tickers).
        """
        self.agent = agent
        self._screen_mode = screen_mode
        
        # Keep original tickers for stock agents (don't add bond ETFs to avoid unnecessary analysis)
        self.tickers = tickers
        
        # Separate bond ETF handling when fixed-income agent is selected
        self.include_fixed_income = "macro_analyst" in selected_analysts or "forward_looking_analyst" in selected_analysts
        self.bond_etfs = ["SHY", "TLT"] if self.include_fixed_income else []
        
        # Combined list for portfolio initialization and price fetching
        self.all_tickers = tickers + self.bond_etfs
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.model_name = model_name
        self.model_provider = model_provider
        self.selected_analysts = selected_analysts
        self.debug_mode = debug_mode
        self.verbose_data = verbose_data

        # Store the margin ratio (e.g. 0.5 means 50% margin required).
        self.margin_ratio = initial_margin_requirement

        # Initialize portfolio with support for long/short positions (stocks + bond ETFs)
        self.portfolio_values = []
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
                } for ticker in self.all_tickers  # Include both stocks and bond ETFs
            },
            "realized_gains": {
                ticker: {
                    "long": 0.0,   # Realized gains from long positions
                    "short": 0.0,  # Realized gains from short positions
                } for ticker in self.all_tickers  # Include both stocks and bond ETFs
            }
        }

    def execute_trade(self, ticker: str, action: str, quantity: float, execution_price: float):
        """
        Execute trades with support for both long and short positions.
        `quantity` is the number of shares the agent wants to buy/sell/short/cover.
        We will only trade integer shares to keep it simple.
        """
        if quantity <= 0:
            return 0

        quantity = int(quantity)  # force integer shares
        position = self.portfolio["positions"][ticker]

        if action == "buy":
            cost = quantity * execution_price
            if cost <= self.portfolio["cash"]:
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
                self.portfolio["cash"] -= cost
                return quantity
            else:
                # Calculate maximum affordable quantity
                max_quantity = int(self.portfolio["cash"] / execution_price)
                if max_quantity > 0:
                    cost = max_quantity * execution_price
                    old_shares = position["long"]
                    old_cost_basis = position["long_cost_basis"]
                    total_shares = old_shares + max_quantity

                    if total_shares > 0:
                        total_old_cost = old_cost_basis * old_shares
                        total_new_cost = cost
                        position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                    position["long"] += max_quantity
                    self.portfolio["cash"] -= cost
                    return max_quantity
                return 0

        elif action == "sell":
            # You can only sell as many as you own
            quantity = min(quantity, position["long"])
            if quantity > 0:
                # Realized gain/loss using average cost basis
                avg_cost_per_share = position["long_cost_basis"] if position["long"] > 0 else 0
                realized_gain = (execution_price - avg_cost_per_share) * quantity
                self.portfolio["realized_gains"][ticker]["long"] += realized_gain

                position["long"] -= quantity
                self.portfolio["cash"] += quantity * execution_price

                if position["long"] == 0:
                    position["long_cost_basis"] = 0.0

                return quantity

        elif action == "short":
            """
            Typical short sale flow:
              1) Receive proceeds = execution_price * quantity
              2) Post margin_required = proceeds * margin_ratio
              3) Net effect on cash = +proceeds - margin_required
            """
            proceeds = execution_price * quantity
            margin_required = proceeds * self.margin_ratio
            if margin_required <= self.portfolio["cash"]:
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

                # Increase cash by proceeds, then subtract the required margin
                self.portfolio["cash"] += proceeds
                self.portfolio["cash"] -= margin_required
                return quantity
            else:
                # Calculate maximum shortable quantity
                if self.margin_ratio > 0:
                    max_quantity = int(self.portfolio["cash"] / (execution_price * self.margin_ratio))
                else:
                    max_quantity = 0

                if max_quantity > 0:
                    proceeds = execution_price * max_quantity
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
                    self.portfolio["cash"] -= margin_required
                    return max_quantity
                return 0

        elif action == "cover":
            """
            When covering shares:
              1) Pay cover cost = execution_price * quantity
              2) Release a proportional share of the margin
              3) Net effect on cash = -cover_cost + released_margin
            """
            quantity = min(quantity, position["short"])
            if quantity > 0:
                cover_cost = quantity * execution_price
                avg_short_price = position["short_cost_basis"] if position["short"] > 0 else 0
                realized_gain = (avg_short_price - execution_price) * quantity

                if position["short"] > 0:
                    portion = quantity / position["short"]
                else:
                    portion = 1.0

                margin_to_release = portion * position["short_margin_used"]

                position["short"] -= quantity
                position["short_margin_used"] -= margin_to_release
                self.portfolio["margin_used"] -= margin_to_release

                # Pay the cost to cover, but get back the released margin
                self.portfolio["cash"] += margin_to_release
                self.portfolio["cash"] -= cover_cost

                self.portfolio["realized_gains"][ticker]["short"] += realized_gain

                if position["short"] == 0:
                    position["short_cost_basis"] = 0.0
                    position["short_margin_used"] = 0.0

                return quantity

        return 0

    def calculate_portfolio_value(self, evaluation_prices):
        """
        Calculate total portfolio value, including:
          - cash
          - market value of long positions
          - unrealized gains/losses for short positions
        """
        total_value = self.portfolio["cash"]

        for ticker in self.all_tickers:
            position = self.portfolio["positions"][ticker]
            current_price = evaluation_prices[ticker]

            # Long positions: Add current market value
            if position["long"] > 0:
                long_market_value = position["long"] * current_price
                total_value += long_market_value

            # Short position unrealized PnL = short_shares * (short_cost_basis - evaluation_prices)
            if position["short"] > 0:
                short_unrealized_pnl = position["short"] * (position["short_cost_basis"] - current_price)
                total_value += short_unrealized_pnl

        return total_value

    def prefetch_data(self):
        """Pre-fetch all data needed for the backtest period."""

        logger.info("Pre-fetching data for the entire backtest period...", module="prefetch_data")

        # Convert string dates to datetime objects
        start_date_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(self.end_date, "%Y-%m-%d")

        # Check if the start and end dates are trading days
        start_valid = is_trading_day(start_date_dt)
        end_valid = is_trading_day(end_date_dt)

        # Notify if start and end date are not trading days with suggestion for other dates.
        if not start_valid:
            next_day = get_next_trading_day(start_date_dt)
            prev_day = get_previous_trading_day(start_date_dt)
            logger.info(f"Start date {start_date_dt} is not a trading day. Consider using {prev_day} or {next_day} instead.", module="prefetch_data")
            
            # Automatically adjust to next trading day instead of exiting
            self.start_date = prev_day
            logger.info(f"Automatically adjusted start date to previous trading day: {self.start_date}", module="prefetch_data")

    
        if not end_valid:
            next_day = get_next_trading_day(end_date_dt)
            prev_day = get_previous_trading_day(end_date_dt)
            logger.info(f"End date {end_date_dt} is not a trading day. Consider using {prev_day} or {next_day} instead.", module="prefetch_data")
            
            # Automatically adjust to previous trading day instead of exiting
            self.end_date = prev_day
            logger.info(f"Automatically adjusted end date to previous trading day: {self.end_date}", module="prefetch_data")
            
        # Fetch up to one year worth of data before start_date
        adjusted_start_date_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        historical_start_dt = adjusted_start_date_dt - relativedelta(years=1)
        historical_start_str = historical_start_dt.strftime("%Y-%m-%d")

        logger.info(f"Fetching historical data from {historical_start_str} to {self.end_date}", module="prefetch_data")

        # Get all potential tickers that could be eligible during the backtest period
        # This includes tickers that might not be eligible at the start but become eligible later
        if self._screen_mode:
            # When screening, get all potential tickers from WRDS that could be eligible
            from utils.tickers import get_sp500_tickers_with_dates
            ticker_dates = get_sp500_tickers_with_dates(years_back=4)
            all_potential_tickers = list(ticker_dates.keys())
            logger.info(f"Prefetching data for {len(all_potential_tickers)} potential S&P 500 tickers", module="prefetch_data")
        else:
            # When running specific tickers, use the provided list
            all_potential_tickers = self.tickers
            logger.info(f"Prefetching data for {len(all_potential_tickers)} specified tickers", module="prefetch_data")

        # Fetch data for all potential stocks
        data = get_data_for_tickers(all_potential_tickers, historical_start_str, self.end_date, verbose_data=self.verbose_data)
        
        # Fetch data for fixed-income assets and FRED macroeconomic data
        if self.include_fixed_income:
            logger.info("Prefetching FRED data and bond ETFs", module="prefetch_data")
            
            # Prefetch bond ETF price data
            for bond_etf in self.bond_etfs:
                get_price_data(bond_etf, historical_start_str, self.end_date, verbose_data=self.verbose_data)

            # Prefetch all FRED macroeconomic data
            try:
                from tools.fred_api import get_all_fred_data_for_agents
                get_all_fred_data_for_agents(historical_start_str, self.end_date, verbose_data=self.verbose_data)
            except Exception as e:
                logger.warning(f"Failed to prefetch FRED data: {e}", module="prefetch_data")
                logger.warning("Agents will fetch FRED data on-demand during backtesting", module="prefetch_data")

        # Log a summary of fetched data
        logger.info(f"Checking verbose_data condition: self.verbose_data = {self.verbose_data}", module="prefetch_data")
        if self.verbose_data:
            logger.debug("=== DATA FETCHED SUMMARY from prefetch_data ===", module="prefetch_data")
            logger.debug(f"Stock tickers: {self.tickers}", module="prefetch_data")
            logger.debug(f"Bond ETFs: {self.bond_etfs}", module="prefetch_data")
            logger.debug(f"Period: {self.start_date} to {self.end_date}", module="prefetch_data")
            
            for ticker in self.all_tickers:
                ticker_data = data.get(ticker, {})
                prices = ticker_data.get("prices")
                metrics = ticker_data.get("metrics", [])
                insider_trades = ticker_data.get("insider_trades", [])
                news = ticker_data.get("news", [])
                
                logger.debug(f"Data structure for {ticker}:", module="prefetch_data")
                logger.debug(f"  - prices: DataFrame with {prices.shape[0] if hasattr(prices, 'shape') else 0} rows, {prices.shape[1] if hasattr(prices, 'shape') else 0} columns", module="prefetch_data")
                logger.debug(f"  - metrics: List with {len(metrics)} items", module="prefetch_data")
                logger.debug(f"  - insider_trades: List with {len(insider_trades)} items", module="prefetch_data")
                logger.debug(f"  - news: List with {len(news)} items", module="prefetch_data")
                
                
                if hasattr(prices, 'head'):
                    logger.debug(f"\nPrices sample:\n{prices.head()}", module="prefetch_data", ticker=ticker)
                
                if metrics and len(metrics) > 0:
                    logger.debug(f"\nMetrics sample:\n {metrics[0].model_dump()}", module="prefetch_data", ticker=ticker)

        logger.info("Data pre-fetch complete.", module="prefetch_data")

    def parse_agent_response(self, agent_output):
        """Parse JSON output from the agent (fallback to 'hold' if invalid)."""
        import json

        try:
            decision = json.loads(agent_output)
            return decision
        except Exception:
            print(f"Error parsing action: {agent_output}")
            return {"action": "hold", "quantity": 0}

    def run_backtest(self):
        from utils.logger import logger

        # Pre-fetch all data at the start
        self.prefetch_data()
        
        # Filter out tickers with no price data for the analysis period
        self._filter_available_tickers()

        dates = pd.date_range(self.start_date, self.end_date, freq="B")
        table_rows = []
        performance_metrics = {
            'sharpe_ratio': None,
            'sortino_ratio': None,
            'max_drawdown': None,
            'long_short_ratio': None,
            'gross_exposure': None,
            'net_exposure': None
        }

        logger.info("Starting backtest...", module="run_backtest")

        # Initialize portfolio values list with initial capital
        if len(dates) > 0:
            self.portfolio_values = [{"Date": dates[0], "Portfolio Value": self.initial_capital}]
        else:
            self.portfolio_values = []

        for current_date in dates:

            logger.info(f"Running hedge fund for {current_date.strftime("%Y-%m-%d")}", module="run_backtest")

            lookback_start = (current_date - timedelta(days=200)).strftime("%Y-%m-%d") # Half year lookback period plus some slack
            current_date_str = current_date.strftime("%Y-%m-%d")
            previous_date_str = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")

            # Skip if there's no prior day to look back (i.e., first date in the range)
            if lookback_start == current_date_str:
                logger.warning("There is no prior day to look back (i.e., first date in the range)", module="run_backtest")
                continue

            # Get eligible tickers for this specific date
            if self._screen_mode:
                from utils.tickers import get_eligible_tickers_for_date
                eligible_tickers = get_eligible_tickers_for_date(current_date_str, min_days_listed=200, years_back=4)
                logger.debug(f"Using {len(eligible_tickers)} eligible tickers for {current_date_str}", module="run_backtest")
            else:
                # Use the original ticker list for non-screening mode
                eligible_tickers = self.tickers

            # Get current prices for all tickers

            try:
                execution_prices = {} # Prices for trade execution
                evaluation_prices = {} # Prices for portfolio value evaluation
                
                # Fetch prices for all tickers (stocks + bond ETFs) for execution and portfolio evaluation
                for ticker in self.all_tickers:

                    price_df = get_price_data(ticker, previous_date_str, current_date_str, self.verbose_data)
                    if self.verbose_data:
                        logger.debug(f"price_df: {price_df}", module="run_backtest", ticker=ticker)

                    try:
                        if len(price_df) < 2:
                            logger.warning(f"Insufficient price data for {ticker} from {previous_date_str} to {current_date_str}. Need at least 2 trading days.", module="run_backtest")
                        

                        # Use current trading day's open for trade execution
                        execution_price = price_df.iloc[-1]["open"]
                        # Use the current trading day's close to evaluate the portfolio value
                        evaluation_price = price_df.iloc[-1]["close"]

                        execution_prices[ticker] = execution_price
                        evaluation_prices[ticker] = evaluation_price

                        if verbose_data:
                            logger.debug(f"Execution price (current open): {execution_prices[ticker]}", module="run_backtest", ticker=ticker)
                            logger.debug(f"Evaluation price (current close): {evaluation_prices[ticker]}", module="run_backtest", ticker=ticker)

                    except:
                        logger.warning(f"Using fallback method for prices on prev_date: {previous_date_str} and current date: {current_date_str}", module="run_backtest")

                        price_df = get_price_data(ticker, previous_date_str, current_date_str, self.verbose_data)

                        evaluation_price = price_df.iloc[-1]["close"]

                        execution_prices[ticker] = evaluation_price
                        evaluation_prices[ticker] = evaluation_price   

            except Exception as e:
                # If data is missing or there's an API error, skip this day
                logger.error(f"Error fetching prices between {previous_date_str} and {current_date_str}: {e}", module="run_backtest")
                continue

            # ---------------------------------------------------------------
            # 1) Execute the agent's trades
            # ---------------------------------------------------------------
            
            logger.info(f"Executing trades on {current_date}", module="run_backtest")

            output = self.agent( 
                tickers=eligible_tickers,
                start_date=lookback_start, # Review: Is only needed by technicals and risk manager agent?
                end_date=current_date_str,
                portfolio=self.portfolio,
                model_name=self.model_name,
                model_provider=self.model_provider,
                selected_analysts=self.selected_analysts,
                verbose_data=self.verbose_data
            )

            decisions = output["decisions"]
            analyst_signals = output["analyst_signals"]

            
            logger.debug(f"Agent decisions for {current_date_str}:", module="run_backtest")
            for ticker, decision in decisions.items():
                logger.debug(f"  {ticker}: {decision.get('action', 'hold')} {decision.get('quantity', 0)} shares", 
                            module="run_backtest", 
                            ticker=ticker)

            # Execute trades for each ticker (stocks + bond ETFs)
            executed_trades = {}
            for ticker in self.all_tickers:
                decision = decisions.get(ticker, {"action": "hold", "quantity": 0})
                action, quantity = decision.get("action", "hold"), decision.get("quantity", 0)

                executed_quantity = self.execute_trade(ticker, action, quantity, execution_prices[ticker])
                executed_trades[ticker] = executed_quantity

            # ---------------------------------------------------------------
            # 2) Now that trades have been executed, recalculate the final
            #    portfolio value for this day.
            # ---------------------------------------------------------------

            total_value = self.calculate_portfolio_value(evaluation_prices)

            # Also compute long/short exposures for final post‐trade state
            long_exposure = sum(
                self.portfolio["positions"][t]["long"] * evaluation_prices[t]
                for t in self.all_tickers
            )
            short_exposure = sum(
                self.portfolio["positions"][t]["short"] * evaluation_prices[t]
                for t in self.all_tickers
            )

            # Calculate gross and net exposures
            gross_exposure = long_exposure + short_exposure
            net_exposure = long_exposure - short_exposure
            long_short_ratio = (
                long_exposure / short_exposure if short_exposure > 1e-9 else float('inf')
            )

            # Track each day's portfolio value in self.portfolio_values
            self.portfolio_values.append({
                "Date": current_date,
                "Portfolio Value": total_value,
                "Long Exposure": long_exposure,
                "Short Exposure": short_exposure,
                "Gross Exposure": gross_exposure,
                "Net Exposure": net_exposure,
                "Long/Short Ratio": long_short_ratio
            })

            # ---------------------------------------------------------------
            # 3) Build the table rows to display
            # ---------------------------------------------------------------
            date_rows = []

            # For each ticker, record signals/trades
            for ticker in eligible_tickers:
                ticker_signals = {}
                for agent_name, signals in analyst_signals.items():
                    if ticker in signals:
                        ticker_signals[agent_name] = signals[ticker]

                bullish_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "bullish"])
                bearish_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "bearish"])
                neutral_count = len([s for s in ticker_signals.values() if s.get("signal", "").lower() == "neutral"])

                # Calculate net position value
                pos = self.portfolio["positions"][ticker]
                long_val = pos["long"] * evaluation_prices[ticker]
                short_val = pos["short"] * evaluation_prices[ticker]
                net_position_value = long_val - short_val

                # Get the action and quantity from the decisions
                action = decisions.get(ticker, {}).get("action", "hold")
                quantity = executed_trades.get(ticker, 0)
                
                # Append the agent action to the table rows
                date_rows.append(
                    format_backtest_row(
                        date=current_date_str,
                        ticker=ticker,
                        action=action,
                        quantity=quantity,
                        open_price=execution_prices[ticker],
                        close_price=evaluation_prices[ticker],
                        shares_owned=pos["long"] - pos["short"],  # net shares
                        position_value=net_position_value,
                        bullish_count=bullish_count,
                        bearish_count=bearish_count,
                        neutral_count=neutral_count,
                    )
                )
            
            # Add bond ETF positions to display (no analyst signals for bonds)
            for bond_etf in self.bond_etfs:
                pos = self.portfolio["positions"][bond_etf]
                long_val = pos["long"] * evaluation_prices[bond_etf]
                short_val = pos["short"] * evaluation_prices[bond_etf]
                net_position_value = long_val - short_val
                
                # Get the action and quantity from the decisions
                action = decisions.get(bond_etf, {}).get("action", "hold")
                quantity = executed_trades.get(bond_etf, 0)
                
                date_rows.append(
                    format_backtest_row(
                        date=current_date_str,
                        ticker=bond_etf,
                        action=action,
                        quantity=quantity,
                        open_price=execution_prices[bond_etf],
                        close_price=evaluation_prices[bond_etf],
                        shares_owned=pos["long"] - pos["short"],  # net shares
                        position_value=net_position_value,
                        bullish_count=0,  # No analyst signals for bonds
                        bearish_count=0,
                        neutral_count=0,
                    )
                )
            # ---------------------------------------------------------------
            # 4) Calculate performance summary metrics
            # ---------------------------------------------------------------
            total_realized_gains = sum(
                self.portfolio["realized_gains"][t]["long"] +
                self.portfolio["realized_gains"][t]["short"]
                for t in self.all_tickers
            )

            # Calculate cumulative return vs. initial capital
            portfolio_return = ((total_value + total_realized_gains) / self.initial_capital - 1) * 100

            # Add summary row for this day
            date_rows.append(
                format_backtest_row(
                    date=current_date_str,
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
                    total_value=total_value,
                    return_pct=portfolio_return,
                    cash_balance=self.portfolio["cash"],
                    total_position_value=total_value - self.portfolio["cash"],
                    sharpe_ratio=performance_metrics["sharpe_ratio"],
                    sortino_ratio=performance_metrics["sortino_ratio"],
                    max_drawdown=performance_metrics["max_drawdown"],
                ),
            )

            table_rows.extend(date_rows)
            print_backtest_results(table_rows)

            # Update performance metrics if we have enough data
            if len(self.portfolio_values) > 3:
                self._update_performance_metrics(performance_metrics)

        return performance_metrics

    def _update_performance_metrics(self, performance_metrics):
        """Helper method to update performance metrics using daily returns."""
        values_df = pd.DataFrame(self.portfolio_values).set_index("Date")
        values_df["Daily Return"] = values_df["Portfolio Value"].pct_change()
        clean_returns = values_df["Daily Return"].dropna()

        if len(clean_returns) < 2:
            return  # not enough data points

        # Assumes 252 trading days/year
        daily_risk_free_rate = 0.0434 / 252
        excess_returns = clean_returns - daily_risk_free_rate
        mean_excess_return = excess_returns.mean()
        std_excess_return = excess_returns.std()

        # Sharpe ratio
        if std_excess_return > 1e-12:
            performance_metrics["sharpe_ratio"] = np.sqrt(252) * (mean_excess_return / std_excess_return)
        else:
            performance_metrics["sharpe_ratio"] = 0.0

        # Sortino ratio
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std()
            if downside_std > 1e-12:
                performance_metrics["sortino_ratio"] = np.sqrt(252) * (mean_excess_return / downside_std)
            else:
                performance_metrics["sortino_ratio"] = float('inf') if mean_excess_return > 0 else 0
        else:
            performance_metrics["sortino_ratio"] = float('inf') if mean_excess_return > 0 else 0

        # Maximum drawdown
        rolling_max = values_df["Portfolio Value"].cummax()
        drawdown = (values_df["Portfolio Value"] - rolling_max) / rolling_max
        performance_metrics["max_drawdown"] = drawdown.min() * 100

    def analyze_performance(self):
        """Creates a performance DataFrame, prints summary stats, and plots equity curve."""
        if not self.portfolio_values:
            print("No portfolio data found. Please run the backtest first.")
            return pd.DataFrame()

        performance_df = pd.DataFrame(self.portfolio_values).set_index("Date")
        if performance_df.empty:
            print("No valid performance data to analyze.")
            return performance_df

        final_portfolio_value = performance_df["Portfolio Value"].iloc[-1]
        total_realized_gains = sum(
            self.portfolio["realized_gains"][ticker]["long"] for ticker in self.all_tickers
        )
        total_return = ((final_portfolio_value - self.initial_capital) / self.initial_capital) * 100

        print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO PERFORMANCE SUMMARY:{Style.RESET_ALL}")
        print(f"Total Return: {Fore.GREEN if total_return >= 0 else Fore.RED}{total_return:.2f}%{Style.RESET_ALL}")
        print(f"Total Realized Gains/Losses: {Fore.GREEN if total_realized_gains >= 0 else Fore.RED}${total_realized_gains:,.2f}{Style.RESET_ALL}")

        # Plot the portfolio value over time
        plt.figure(figsize=(12, 6))
        plt.plot(performance_df.index, performance_df["Portfolio Value"], color="blue")
        plt.title("Portfolio Value Over Time")
        plt.ylabel("Portfolio Value ($)")
        plt.xlabel("Date")
        plt.grid(True)
        plt.show(block=False)

        # Compute daily returns
        performance_df["Daily Return"] = performance_df["Portfolio Value"].pct_change().fillna(0)
        daily_rf = 0.0434 / 252  # daily risk-free rate
        mean_daily_return = performance_df["Daily Return"].mean()
        std_daily_return = performance_df["Daily Return"].std()

        # Annualized Sharpe Ratio
        if std_daily_return != 0:
            annualized_sharpe = np.sqrt(252) * ((mean_daily_return - daily_rf) / std_daily_return)
        else:
            annualized_sharpe = 0
        print(f"\nAnnualized Sharpe Ratio: {Fore.YELLOW}{annualized_sharpe:.2f}{Style.RESET_ALL}")

        # Max Drawdown
        rolling_max = performance_df["Portfolio Value"].cummax()
        drawdown = (performance_df["Portfolio Value"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin()
        if pd.notnull(max_drawdown_date):
            print(f"Maximum Drawdown: {Fore.RED}{max_drawdown * 100:.2f}%{Style.RESET_ALL} (on {max_drawdown_date.strftime('%Y-%m-%d')})")
        else:
            print(f"Maximum Drawdown: {Fore.RED}0.00%{Style.RESET_ALL}")

        # Win Rate
        winning_days = len(performance_df[performance_df["Daily Return"] > 0])
        total_days = max(len(performance_df) - 1, 1)
        win_rate = (winning_days / total_days) * 100
        print(f"Win Rate: {Fore.GREEN}{win_rate:.2f}%{Style.RESET_ALL}")

        # Average Win/Loss Ratio
        positive_returns = performance_df[performance_df["Daily Return"] > 0]["Daily Return"]
        negative_returns = performance_df[performance_df["Daily Return"] < 0]["Daily Return"]
        avg_win = positive_returns.mean() if not positive_returns.empty else 0
        avg_loss = abs(negative_returns.mean()) if not negative_returns.empty else 0
        if avg_loss != 0:
            win_loss_ratio = avg_win / avg_loss
        else:
            win_loss_ratio = float('inf') if avg_win > 0 else 0
        print(f"Win/Loss Ratio: {Fore.GREEN}{win_loss_ratio:.2f}{Style.RESET_ALL}")

        # Maximum Consecutive Wins / Losses
        returns_binary = (performance_df["Daily Return"] > 0).astype(int)
        if len(returns_binary) > 0:
            max_consecutive_wins = max((len(list(g)) for k, g in itertools.groupby(returns_binary) if k == 1), default=0)
            max_consecutive_losses = max((len(list(g)) for k, g in itertools.groupby(returns_binary) if k == 0), default=0)
        else:
            max_consecutive_wins = 0
            max_consecutive_losses = 0

        print(f"Max Consecutive Wins: {Fore.GREEN}{max_consecutive_wins}{Style.RESET_ALL}")
        print(f"Max Consecutive Losses: {Fore.RED}{max_consecutive_losses}{Style.RESET_ALL}")

        return performance_df


### 4. Run the Backtest #####
if __name__ == "__main__":
    import argparse
    from utils.logger import setup_logger, logger

    parser = argparse.ArgumentParser(description="Run backtesting simulation")
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of stock ticker symbols (e.g., AAPL,MSFT,GOOGL)",
    )
    ticker_group.add_argument(
        "--screen",
        action="store_true",
        help="Screen all S&P 500 tickers instead of specifying individual tickers"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - relativedelta(months=1)).strftime("%Y-%m-%d"),
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000,
        help="Initial capital amount (default: 100000)",
    )
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=0.0,
        help="Margin ratio for short positions, e.g. 0.5 for 50 percent (default: 0.0)",
    )

    # Debugging options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging",
    )
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        help="Save logs to a file in the logs directory",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Specify a custom log file path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimize output (overrides debug mode)",
    )
    parser.add_argument(
        "--verbose-data",
        action="store_true",
        help="Show detailed data output (works with debug mode)",
    )

    args = parser.parse_args()

    # Set up the logger with command line arguments
    setup_logger( # Review: What happens if I would delete this setup_logger? What does it do?
        debug_mode=args.debug and not args.quiet, # Review: explain this line
        log_to_file=args.log_to_file,
        log_file=args.log_file
    )

    # Set a global flag for verbose data output
    verbose_data = args.verbose_data and args.debug and not args.quiet

    # Parse tickers based on the selected mode
    if args.screen:
        tickers = get_sp500_tickers()
        logger.info(f"Using {len(tickers)} S&P 500 tickers for screening (source: Wikipedia)", module="backtester")
    else:
        # Parse tickers from comma-separated string
        tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    # Choose analysts
    selected_analysts = None
    choices = questionary.checkbox(
        "Use the Space bar to select/unselect analysts.",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\n\nPress 'a' to toggle all.\n\nPress Enter when done to run the hedge fund.",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        selected_analysts = choices
        print(
            f"\nSelected analysts: "
            f"{', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}"
        )

    # Select LLM model
    model_choice = questionary.select(
        "Select your LLM model:",
        choices=[questionary.Choice(display, value=value) for display, value, _ in LLM_ORDER],
        style=questionary.Style([
            ("selected", "fg:green bold"),
            ("pointer", "fg:green bold"),
            ("highlighted", "fg:green"),
            ("answer", "fg:green bold"),
        ])
    ).ask()

    if not model_choice:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        model_info = get_model_info(model_choice)
        if model_info:
            model_provider = model_info.provider.value
            print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
        else:
            model_provider = "Unknown"
            print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")

    
    # Create the backtester
    backtester = Backtester(
        agent=run_hedge_fund,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        model_name=model_choice,
        model_provider=model_provider,
        selected_analysts=selected_analysts,
        initial_margin_requirement=args.margin_requirement,
        debug_mode=args.debug and not args.quiet,
        verbose_data=args.verbose_data and args.debug and not args.quiet,
        screen_mode=args.screen,
    )

    # Start the timer after LLM and analysts are selected
    start_time = time.time()

    # Run the backtester
    performance_metrics = backtester.run_backtest()
    performance_df = backtester.analyze_performance()

    # Stop timer after execution
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n⏱ Total elapsed time: {int(hours)}h {int(minutes)}m {seconds:.2f}s.")
