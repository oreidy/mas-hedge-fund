from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import json

from tools.api import get_financial_metrics, get_market_cap, search_line_items
from utils.logger import logger


##### Valuation Agent #####
def valuation_agent(state: AgentState):
    """Performs detailed valuation analysis using multiple methodologies for multiple tickers."""

    # Get verbose_data from metadata or default to False
    verbose_data = state["metadata"].get("verbose_data", False)
    logger.debug("Accessing Valuation Agent", module="valuation_agent")

    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize valuation analysis for each ticker
    valuation_analysis = {}

    for ticker in tickers:
        progress.update_status("valuation_agent", ticker, "Fetching financial data")

        # Fetch the financial metrics
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="annual",
            verbose_data=verbose_data
        )

        # Add safety check for financial metrics
        if not financial_metrics:
            logger.warning(f"No financial metrics found for {ticker}", module="valuation_agent", ticker=ticker)
            progress.update_status("valuation_agent", ticker, "Failed: No financial metrics found")
            continue
        
        metrics = financial_metrics[0]

        # Cap earnings growth to prevent extreme valuations
        raw_earnings_growth = metrics.earnings_growth
        if raw_earnings_growth is not None:
            # Cap growth between -95% and +200% to prevent formula explosion
            capped_earnings_growth = max(-0.95, min(raw_earnings_growth, 2.0))
            if capped_earnings_growth != raw_earnings_growth:
                logger.debug(f"Capped extreme earnings growth for {ticker}: {raw_earnings_growth:.1%} -> {capped_earnings_growth:.1%}", 
                             module="valuation_agent", ticker=ticker)
            metrics.earnings_growth = capped_earnings_growth

        if verbose_data:
            logger.debug(f"Retrieved financial metrics for {ticker}:", module="valuation_agent", ticker=ticker)
            logger.debug(f"Earnings growth: {metrics.earnings_growth}", module="valuation_agent", ticker=ticker)
            logger.debug(f"P/E ratio: {metrics.price_to_earnings_ratio}", module="valuation_agent", ticker=ticker)

        progress.update_status("valuation_agent", ticker, "Gathering line items")
        # Fetch the specific line_items that we need for valuation purposes
        financial_line_items = search_line_items(
            ticker=ticker,
            line_items=[
                "free_cash_flow",
                "net_income",
                "depreciation_and_amortization",
                "capital_expenditure",
                "working_capital",
            ],
            end_date=end_date,
            period="annual",
            limit=2,
            verbose_data=verbose_data
        )

        # Add safety check for financial line items
        if len(financial_line_items) < 2:
            logger.warning(f"Insufficient financial line items for {ticker}: only {len(financial_line_items)} periods found, need at least 2", 
                         module="valuation_agent", ticker=ticker)
            if financial_line_items:
                logger.debug(f"Retrieved line items: {[item.model_dump() for item in financial_line_items]}", 
                           module="valuation_agent", ticker=ticker)
            progress.update_status("valuation_agent", ticker, "Failed: Insufficient financial line items")
            continue

        # Pull the current and previous financial line items
        current_financial_line_item = financial_line_items[0]
        previous_financial_line_item = financial_line_items[1]

        if verbose_data:
            logger.debug(f"Current financial line item:", module="valuation_agent", ticker=ticker)
            logger.debug(f"  - Report period: {current_financial_line_item.report_period}", module="valuation_agent", ticker=ticker)
            logger.debug(f"  - Free cash flow: {current_financial_line_item.free_cash_flow}", module="valuation_agent", ticker=ticker)
            logger.debug(f"  - Net income: {current_financial_line_item.net_income}", module="valuation_agent", ticker=ticker)
            logger.debug(f"  - Working capital: {current_financial_line_item.working_capital}", module="valuation_agent", ticker=ticker)


        progress.update_status("valuation_agent", ticker, "Calculating owner earnings")
        
        # Get critical financial metrics with safe attribute access
        net_income = getattr(current_financial_line_item, 'net_income', None)
        free_cash_flow = getattr(current_financial_line_item, 'free_cash_flow', None)
        
        # Skip valuation if net income is unavailable
        if net_income is None:
            logger.warning(f"Net income unavailable for {ticker}, cannot perform valuation analysis", module="valuation_agent", ticker=ticker)
            valuation_analysis[ticker] = {
                "signal": "hold",
                "confidence": 0,
                "reasoning": {"error": "Insufficient financial data - net income unavailable"}
            }
            progress.update_status("valuation_agent", ticker, "Skipped: No net income data")
            continue
            
        # Calculate working capital change
        current_wc = getattr(current_financial_line_item, 'working_capital', None)
        previous_wc = getattr(previous_financial_line_item, 'working_capital', None)
        
        # Track if we're using estimated data for confidence adjustment
        using_estimated_data = False
        
        if current_wc is None or previous_wc is None:
            logger.warning(f"Working capital data unavailable for {ticker}, using 0 for calculation", module="valuation_agent", ticker=ticker)
            working_capital_change = 0
            using_estimated_data = True
        else:
            working_capital_change = current_wc - previous_wc

        # Get other financial metrics with safe attribute access
        depreciation = getattr(current_financial_line_item, 'depreciation_and_amortization', None)
        capex = getattr(current_financial_line_item, 'capital_expenditure', None)
        
        # Handle missing optional financial data
        if depreciation is None:
            logger.warning(f"Depreciation data unavailable for {ticker}, using 0 for calculation", module="valuation_agent", ticker=ticker)
            depreciation = 0
            using_estimated_data = True
            
        if capex is None:
            logger.warning(f"Capital expenditure data unavailable for {ticker}, using 0 for calculation", module="valuation_agent", ticker=ticker)
            capex = 0
            using_estimated_data = True

        # Owner Earnings Valuation (Buffett Method)
        owner_earnings_value = calculate_owner_earnings_value(
            net_income=net_income,
            depreciation=depreciation,
            capex=capex,
            working_capital_change=working_capital_change,
            growth_rate=metrics.earnings_growth,
            required_return=0.15,
            margin_of_safety=0.25,
        )

        if verbose_data:
            logger.debug(f"Owner earnings value: {owner_earnings_value}", module="valuation_agent", ticker=ticker)
                

        progress.update_status("valuation_agent", ticker, "Calculating DCF value")
        
        # Handle missing free cash flow for DCF
        if free_cash_flow is None:
            logger.warning(f"Free cash flow data unavailable for {ticker}, using 0 for calculation", module="valuation_agent", ticker=ticker)
            free_cash_flow = 0
            using_estimated_data = True
        
        # DCF Valuation
        dcf_value = calculate_intrinsic_value(
            free_cash_flow=free_cash_flow,
            growth_rate=metrics.earnings_growth,
            discount_rate=0.10,
            terminal_growth_rate=0.03,
            num_years=5,
        )

        if verbose_data:
            logger.debug(f"DCF value: {dcf_value}", module="valuation_agent", ticker=ticker)

        progress.update_status("valuation_agent", ticker, "Comparing to market value")
        # Get the market cap
        market_cap = get_market_cap(ticker=ticker, end_date=end_date, verbose_data=verbose_data)

        if market_cap is None:
            logger.warning(f"No market cap found for {ticker}", module="valuation_agent", ticker=ticker)
            progress.update_status("valuation_agent", ticker, "Failed: No market cap found")
            continue
            
        if verbose_data:
            logger.debug(f"Market cap: {market_cap}", module="valuation_agent", ticker=ticker)

        # Calculate combined valuation gap (average of both methods)
        dcf_gap = (dcf_value - market_cap) / market_cap
        owner_earnings_gap = (owner_earnings_value - market_cap) / market_cap
        valuation_gap = (dcf_gap + owner_earnings_gap) / 2

        if verbose_data:
                logger.debug(f"DCF gap: {dcf_gap:.2%}", module="valuation_agent", ticker=ticker)
                logger.debug(f"Owner earnings gap: {owner_earnings_gap:.2%}", module="valuation_agent", ticker=ticker)
                logger.debug(f"Valuation gap: {valuation_gap:.2%}", module="valuation_agent", ticker=ticker)

        if valuation_gap > 0.15:  # More than 15% undervalued
            signal = "bullish"
        elif valuation_gap < -0.15:  # More than 15% overvalued
            signal = "bearish"
        else:
            signal = "neutral"

        # Create the reasoning
        reasoning = {}
        reasoning["dcf_analysis"] = {
            "signal": ("bullish" if dcf_gap > 0.15 else "bearish" if dcf_gap < -0.15 else "neutral"),
            "details": f"Intrinsic Value: ${dcf_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {dcf_gap:.1%}",
        }

        reasoning["owner_earnings_analysis"] = {
            "signal": ("bullish" if owner_earnings_gap > 0.15 else "bearish" if owner_earnings_gap < -0.15 else "neutral"),
            "details": f"Owner Earnings Value: ${owner_earnings_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {owner_earnings_gap:.1%}",
        }

        confidence = min(round(abs(valuation_gap), 2) * 100, 100.0)  # Cap at 100%
        
        # Reduce confidence when using estimated data
        # Working capital change affects Owner Earnings calculation accuracy - when missing,
        # we assume 0 impact which may not reflect reality, so reduce confidence by 30%
        if using_estimated_data:
            confidence = max(confidence * 0.7, 10)  # Reduce by 30%, minimum 10%
            logger.debug(f"Confidence reduced due to missing working capital data", module="valuation_agent", ticker=ticker)
        
        valuation_analysis[ticker] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        if verbose_data:
            logger.debug(f"Final valuation signal: {signal}", module="valuation_agent", ticker=ticker)
            logger.debug(f"Confidence: {confidence}%", module="valuation_agent", ticker=ticker)

        progress.update_status("valuation_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(valuation_analysis),
        name="valuation_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(valuation_analysis, "Valuation Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["valuation_agent"] = valuation_analysis

    return {
        "messages": [message],
        "data": data,
    }


def calculate_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5,
) -> float:
    """
    Calculates the intrinsic value using Buffett's Owner Earnings method.

    Owner Earnings = Net Income
                    + Depreciation/Amortization
                    - Capital Expenditures
                    - Working Capital Changes

    Args:
        net_income: Annual net income
        depreciation: Annual depreciation and amortization
        capex: Annual capital expenditures
        working_capital_change: Annual change in working capital
        growth_rate: Expected growth rate
        required_return: Required rate of return (Buffett typically uses 15%)
        margin_of_safety: Margin of safety to apply to final value
        num_years: Number of years to project

    Returns:
        float: Intrinsic value with margin of safety
    """
    if not all([isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]]):
        logger.warning(f"returning 0 owner earnings value because of missing financial metrics", module="valuation_agent")
        return 0

    # Handle None growth rate using a default
    if growth_rate is None:
        growth_rate = 0.05
        logger.debug("Growth rate was None, using default 5%", module="calculate_owner_earnings_value")


    # Calculate initial owner earnings
    owner_earnings = net_income + depreciation - capex - working_capital_change

    if owner_earnings <= 0:
        return 0

    # Project future owner earnings
    future_values = []
    for year in range(1, num_years + 1):
        future_value = owner_earnings * (1 + growth_rate) ** year
        discounted_value = future_value / (1 + required_return) ** year
        future_values.append(discounted_value)

    # Calculate terminal value (using perpetuity growth formula)
    terminal_growth = min(growth_rate, 0.03)  # Cap terminal growth at 3%
    terminal_value = (future_values[-1] * (1 + terminal_growth)) / (required_return - terminal_growth)
    terminal_value_discounted = terminal_value / (1 + required_return) ** num_years

    # Sum all values and apply margin of safety
    intrinsic_value = sum(future_values) + terminal_value_discounted
    value_with_safety_margin = intrinsic_value * (1 - margin_of_safety)

    return value_with_safety_margin


def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """
    Computes the discounted cash flow (DCF) for a given company based on the current free cash flow.
    Use this function to calculate the intrinsic value of a stock.
    """

    # Handle None growth rate using a default
    if growth_rate is None:
        growth_rate = 0.05  
        logger.debug("Growth rate was None, using default 5%", module="calculate_intrinsic_value")

    # Estimate the future cash flows based on the growth rate
    cash_flows = [free_cash_flow * (1 + growth_rate) ** i for i in range(num_years)]

    # Calculate the present value of projected cash flows
    present_values = []
    for i in range(num_years):
        present_value = cash_flows[i] / (1 + discount_rate) ** (i + 1)
        present_values.append(present_value)

    # Calculate the terminal value
    terminal_value = cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    terminal_present_value = terminal_value / (1 + discount_rate) ** num_years

    # Sum up the present values and terminal value
    dcf_value = sum(present_values) + terminal_present_value

    return dcf_value


def calculate_working_capital_change(
    current_working_capital: float,
    previous_working_capital: float,
) -> float:
    """
    Calculate the absolute change in working capital between two periods.
    A positive change means more capital is tied up in working capital (cash outflow).
    A negative change means less capital is tied up (cash inflow).

    Args:
        current_working_capital: Current period's working capital
        previous_working_capital: Previous period's working capital

    Returns:
        float: Change in working capital (current - previous)
    """
    return current_working_capital - previous_working_capital
