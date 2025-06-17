from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from tools.api import get_prices, prices_to_df
import json
from utils.logger import logger



##### Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Controls position sizing based on real-world risk factors and macro allocation for multiple tickers."""

    # Get verbose_data from metadata or default to False
    verbose_data = state["metadata"].get("verbose_data", False)
    logger.debug("Accessing Risk Management Agent", module="risk_management_agent")

    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]
    
    # Get macro allocation from macro agent
    macro_allocation = data["analyst_signals"].get("macro_agent", {})
    stock_allocation = macro_allocation.get("stock_allocation", 0.6)  # Default to 60% stocks
    
    if verbose_data:
        logger.debug(f"Macro agent signals received: {json.dumps(macro_allocation, indent=2)}", module="risk_management_agent")
        if "signal_summary" in macro_allocation:
            signal_summary = macro_allocation["signal_summary"]
            logger.debug(f"Macro signal breakdown - Stocks: {signal_summary.get('stock_signals', 0)}, Bonds: {signal_summary.get('bond_signals', 0)}, Neutral: {signal_summary.get('neutral_signals', 0)}", module="risk_management_agent")
        if "reasoning" in macro_allocation:
            reasoning = macro_allocation["reasoning"]
            for category, details in reasoning.items():
                logger.debug(f"Macro {category} analysis: {details.get('signal', 'N/A')} - {details.get('details', 'No details')}", module="risk_management_agent")
    
    progress.update_status("risk_management_agent", "PORTFOLIO", f"Applying macro allocation: {stock_allocation*100:.0f}% stocks")

    # Initialize risk analysis for each ticker
    risk_analysis = {}
    current_prices = {}  # Store prices here to avoid redundant API calls

    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Analyzing price data")

        prices = get_prices(
            ticker=ticker,
            start_date=data["start_date"],
            end_date=data["end_date"],
            verbose_data=verbose_data
        )

        if not prices:
            progress.update_status("risk_management_agent", ticker, "Failed: No price data found")
            continue

        prices_df = prices_to_df(prices)

        progress.update_status("risk_management_agent", ticker, "Calculating position limits")

        # Calculate portfolio value
        current_price = prices_df["close"].iloc[-1]
        current_prices[ticker] = current_price  # Store the current price

        # Calculate current position value for this ticker
        current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0)

        # Calculate total portfolio value using stored prices
        total_portfolio_value = portfolio.get("cash", 0) + sum(portfolio.get("cost_basis", {}).get(t, 0) for t in portfolio.get("cost_basis", {}))

        # Apply macro allocation constraint: scale individual position limits proportionally
        default_stock_allocation = 0.6  # Default 60% stock allocation
        base_individual_limit_pct = 0.20  # Base 20% per position
        
        # Scale individual position limit based on macro allocation vs default
        # If macro says 40% stocks instead of 60%, scale position limits proportionally
        macro_scaling_factor = stock_allocation / default_stock_allocation
        scaled_individual_limit_pct = base_individual_limit_pct * macro_scaling_factor
        
        # This is the only limit we actually need for position sizing
        macro_constrained_limit = total_portfolio_value * scaled_individual_limit_pct

        # For existing positions, subtract current position value from limit
        remaining_position_limit = macro_constrained_limit - current_position_value

        # Ensure we don't exceed available cash
        max_position_size = min(remaining_position_limit, portfolio.get("cash", 0))
        
        if verbose_data:
            logger.debug(f"Position constraints for {ticker}:", module="risk_management_agent")
            logger.debug(f"  Total portfolio value: ${total_portfolio_value:,.2f}", module="risk_management_agent")
            logger.debug(f"  Macro scaling factor: {macro_scaling_factor:.3f}", module="risk_management_agent")
            logger.debug(f"  Scaled individual limit: {scaled_individual_limit_pct*100:.1f}%", module="risk_management_agent")
            logger.debug(f"  Macro-constrained limit: ${macro_constrained_limit:,.2f}", module="risk_management_agent")
            logger.debug(f"  Current position value: ${current_position_value:,.2f}", module="risk_management_agent")
            logger.debug(f"  Remaining position limit: ${remaining_position_limit:,.2f}", module="risk_management_agent")
            logger.debug(f"  Available cash: ${portfolio.get('cash', 0):,.2f}", module="risk_management_agent")
            logger.debug(f"  Final max position size: ${max_position_size:,.2f}", module="risk_management_agent")

        risk_analysis[ticker] = {
            "remaining_position_limit": float(max_position_size),
            "current_price": float(current_price),
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position": float(current_position_value),
                "macro_constrained_limit": float(macro_constrained_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
                "macro_stock_allocation": float(stock_allocation),
            },
        }

        progress.update_status("risk_management_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(risk_analysis),
        name="risk_management_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }
