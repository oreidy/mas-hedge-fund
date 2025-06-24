from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from tools.api import get_prices, prices_to_df
from utils.risk_metrics import calculate_cvar
from datetime import datetime, timedelta
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
    
    # Extend start_date to get 6 months of trading data for CVaR calculation
    # Approximate 6 months = 130 trading days (22 trading days per month * 6)
    original_start_date = data["start_date"]
    extended_start_date = (datetime.strptime(data["end_date"], "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d")  # ~6 months including weekends/holidays
    
    # Get allocations from macro and forward-looking agents
    macro_allocation = data["analyst_signals"].get("macro_agent", {})
    forward_looking_allocation = data["analyst_signals"].get("forward_looking_agent", {})
    
    # Combine macro and forward-looking allocations (average them)
    macro_stock_allocation = macro_allocation.get("stock_allocation", 0.6)
    forward_looking_stock_allocation = forward_looking_allocation.get("stock_allocation", 0.6)
    
    # Average the two allocations
    stock_allocation = (macro_stock_allocation + forward_looking_stock_allocation) / 2
    
    if verbose_data:
        logger.debug(f"Macro agent signals received: {json.dumps(macro_allocation, indent=2)}", module="risk_management_agent")
        logger.debug(f"Forward-looking agent signals received: {json.dumps(forward_looking_allocation, indent=2)}", module="risk_management_agent")
        logger.debug(f"Combined allocation: Macro {macro_stock_allocation*100:.0f}% stocks, Forward-looking {forward_looking_stock_allocation*100:.0f}% stocks, Final {stock_allocation*100:.0f}% stocks", module="risk_management_agent")
        
        if "signal_summary" in macro_allocation:
            signal_summary = macro_allocation["signal_summary"]
            logger.debug(f"Macro signal breakdown - Stocks: {signal_summary.get('stock_signals', 0)}, Bonds: {signal_summary.get('bond_signals', 0)}, Neutral: {signal_summary.get('neutral_signals', 0)}", module="risk_management_agent")
        if "reasoning" in macro_allocation:
            reasoning = macro_allocation["reasoning"]
            for category, details in reasoning.items():
                logger.debug(f"Macro {category} analysis: {details.get('signal', 'N/A')} - {details.get('details', 'No details')}", module="risk_management_agent")
        
        if "current_vix" in forward_looking_allocation:
            logger.debug(f"Forward-looking VIX analysis: Current VIX {forward_looking_allocation['current_vix']}, {forward_looking_allocation.get('reasoning', 'No reasoning')}", module="risk_management_agent")
    
    progress.update_status("risk_management_agent", "PORTFOLIO", f"Applying combined allocation: {stock_allocation*100:.0f}% stocks")

    # Initialize risk analysis for each ticker
    risk_analysis = {}
    current_prices = {}  # Store prices here to avoid redundant API calls

    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Analyzing price data")

        prices = get_prices(
            ticker=ticker,
            start_date=extended_start_date,
            end_date=data["end_date"],
            verbose_data=verbose_data
        )

        if not prices:
            progress.update_status("risk_management_agent", ticker, "Failed: No price data found")
            continue

        prices_df = prices_to_df(prices)
        
        # Calculate 95% confidence CVaR using the same price data
        progress.update_status("risk_management_agent", ticker, "Calculating CVaR")
        cvar_value = calculate_cvar(prices_df, confidence_level=0.05, verbose_data=verbose_data)
        
        # Transform CVaR to risk score using linear mapping
        # CVaR is typically negative (expected loss), so we take absolute value
        # Map CVaR range [-0.15, 0] to risk score [1.0, 0.0] (higher CVaR loss = higher risk score)
        max_cvar_loss = 0.15  # 15% maximum expected loss
        cvar_risk_score = min(abs(cvar_value) / max_cvar_loss, 1.0)  # Linear mapping, capped at 1.0
        
        if verbose_data:
            logger.debug(f"CVaR analysis for {ticker}:", module="risk_management_agent")
            logger.debug(f"  CVaR (95% confidence): {cvar_value*100:.4f}%", module="risk_management_agent")
            logger.debug(f"  CVaR Risk Score: {cvar_risk_score:.4f}", module="risk_management_agent")

        progress.update_status("risk_management_agent", ticker, "Calculating position limits")

        # Calculate portfolio value
        current_price = prices_df["close"].iloc[-1]
        current_prices[ticker] = current_price  # Store the current price

        # Calculate current position value for this ticker
        current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0)

        # Calculate total portfolio value using stored prices
        total_portfolio_value = portfolio.get("cash", 0) + sum(portfolio.get("cost_basis", {}).get(t, 0) for t in portfolio.get("cost_basis", {}))

        # Apply combined allocation constraint: scale individual position limits proportionally
        default_stock_allocation = 0.6  # Default 60% stock allocation
        base_individual_limit_pct = 0.20  # Base 20% per position
        
        # Scale individual position limit based on combined allocation vs default
        # Combined allocation considers both macro and VIX signals
        allocation_scaling_factor = stock_allocation / default_stock_allocation
        scaled_individual_limit_pct = base_individual_limit_pct * allocation_scaling_factor
        
        # Apply allocation constraint first
        allocation_constrained_limit = total_portfolio_value * scaled_individual_limit_pct

        # Apply CVaR-based risk adjustment to the allocation-constrained limit
        # Higher CVaR risk score reduces position limit (less aggressive than before)
        risk_tolerance = 0.8  # High risk tolerance - CVaR has moderate impact
        max_cvar_reduction = 0.5  # Maximum 50% reduction for highest risk
        cvar_adjustment_factor = 1.0 - (cvar_risk_score * (1 - risk_tolerance) * max_cvar_reduction)
        cvar_adjusted_limit = allocation_constrained_limit * cvar_adjustment_factor

        # For existing positions, subtract current position value from limit
        remaining_position_limit = cvar_adjusted_limit - current_position_value

        # Ensure we don't exceed available cash
        max_position_size = min(remaining_position_limit, portfolio.get("cash", 0))
        
        if verbose_data:
            logger.debug(f"Position constraints for {ticker}:", module="risk_management_agent")
            logger.debug(f"  Total portfolio value: ${total_portfolio_value:,.2f}", module="risk_management_agent")
            logger.debug(f"  Allocation scaling factor: {allocation_scaling_factor:.3f}", module="risk_management_agent")
            logger.debug(f"  Scaled individual limit: {scaled_individual_limit_pct*100:.1f}%", module="risk_management_agent")
            logger.debug(f"  Allocation-constrained limit: ${allocation_constrained_limit:,.2f}", module="risk_management_agent")
            logger.debug(f"  CVaR adjustment factor: {cvar_adjustment_factor:.3f}", module="risk_management_agent")
            logger.debug(f"  CVaR-adjusted limit: ${cvar_adjusted_limit:,.2f}", module="risk_management_agent")
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
                "cvar_adjusted_limit": float(cvar_adjusted_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
                "combined_stock_allocation": float(stock_allocation),
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
