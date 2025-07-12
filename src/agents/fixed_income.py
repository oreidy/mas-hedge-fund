import json
from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from utils.logger import logger
from tools.fred_api import get_fred_data
from tools.api import get_prices, prices_to_df
from datetime import datetime, timedelta


##### Fixed Income Agent #####
def fixed_income_agent(state: AgentState):
    """Makes algorithmic bond investment decisions based on yield curve analysis"""
    
    verbose_data = state["metadata"].get("verbose_data", False)
    logger.debug("Accessing Fixed Income Agent", module="fixed_income_agent")
    
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    end_date = state["data"]["end_date"]
    
    progress.update_status("fixed_income_agent", None, "Analyzing bond allocation from risk manager")
    
    # Get bond allocation from risk manager (cleaner than combining macro + forward-looking)
    risk_signals = analyst_signals.get("risk_management_agent", {})
    
    # Extract stock allocation from any ticker's risk analysis (they all have the same combined allocation)
    stock_allocation = 0.6  # Default fallback
    if risk_signals:
        # Get stock allocation from first ticker's risk analysis
        first_ticker_data = next(iter(risk_signals.values()), {})
        stock_allocation = first_ticker_data.get("reasoning", {}).get("combined_stock_allocation", 0.6)
    
    # Bond allocation is complement of stock allocation
    bond_allocation = 1.0 - stock_allocation
    
    if verbose_data:
        logger.debug(f"Allocation analysis:", module="fixed_income_agent")
        logger.debug(f"  Stock allocation: {stock_allocation*100:.1f}%", module="fixed_income_agent")
        logger.debug(f"  Bond allocation: {bond_allocation*100:.1f}%", module="fixed_income_agent")
    
    progress.update_status("fixed_income_agent", "SHY, TLT", "Fetching yield curve data")
    
    # Get yield curve data for algorithmic decision
    yield_curve_data = get_yield_curve_data(end_date, verbose_data)
    
    if verbose_data:
        logger.debug(f"Yield curve analysis:", module="fixed_income_agent")
        for key, value in yield_curve_data.items():
            logger.debug(f"  {key}: {value}", module="fixed_income_agent")
    
    progress.update_status("fixed_income_agent", "SHY, TLT", "Making bond decisions")
    
    # Algorithmic bond selection based on yield curve
    bond_decisions = make_bond_decisions(
        bond_allocation=bond_allocation,
        yield_curve_data=yield_curve_data,
        portfolio=portfolio,
        end_date=end_date,
        verbose_data=verbose_data
    )
    
    message = HumanMessage(
        content=json.dumps(bond_decisions),
        name="fixed_income_agent"
    )
    
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(bond_decisions, "Fixed Income Agent")
    
    # Add the signal to analyst_signals
    state["data"]["analyst_signals"]["fixed_income_agent"] = bond_decisions
    
    progress.update_status("fixed_income_agent", "SHY, TLT", "Done")
    
    return {
        "messages": state["messages"] + [message],
        "data": state["data"]
    }


def get_yield_curve_data(end_date: str, verbose_data: bool = False) -> dict:
    """Fetch yield curve data from FRED API"""
    
    # Calculate date range (1 week of data to ensure we get the most recent available data)
    # Treasury data may not be published daily, so we fetch a week to capture the latest values
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=7)
    start_date = start_dt.strftime("%Y-%m-%d")
    
    yield_data = {}
    
    try:
        # 2-Year Treasury (short-term rate)
        treasury_2y = get_fred_data("GS2", start_date, end_date, verbose_data)
        if treasury_2y is not None and len(treasury_2y) > 0:
            yield_data["treasury_2y"] = treasury_2y.iloc[-1]["value"]
        else:
            logger.warning("Failed to fetch 2-Year Treasury data from FRED", module="fixed_income_agent")
        
        # 10-Year Treasury (long-term rate)
        treasury_10y = get_fred_data("GS10", start_date, end_date, verbose_data)
        if treasury_10y is not None and len(treasury_10y) > 0:
            yield_data["treasury_10y"] = treasury_10y.iloc[-1]["value"]
        else:
            logger.warning("Failed to fetch 10-Year Treasury data from FRED", module="fixed_income_agent")
        
        # Calculate yield curve slope (10Y - 2Y)
        if "treasury_2y" in yield_data and "treasury_10y" in yield_data:
            yield_data["yield_spread"] = yield_data["treasury_10y"] - yield_data["treasury_2y"]
            
            # Classify yield curve shape with finer granularity
            # Spread thresholds based on historical yield curve analysis
            if yield_data["yield_spread"] > 2.0:
                yield_data["curve_shape"] = "very_steep"        # Extremely steep (>200bp)
            elif yield_data["yield_spread"] > 1.0:
                yield_data["curve_shape"] = "steep"             # Steep (100-200bp)
            elif yield_data["yield_spread"] > 0.25:
                yield_data["curve_shape"] = "normal"            # Normal upward slope (25-100bp)
            elif yield_data["yield_spread"] > -0.25:
                yield_data["curve_shape"] = "flat"              # Nearly flat (-25 to +25bp)
            elif yield_data["yield_spread"] > -1.0:
                yield_data["curve_shape"] = "inverted"          # Moderately inverted (-25 to -100bp)
            else:
                yield_data["curve_shape"] = "deeply_inverted"   # Deeply inverted (<-100bp)
        
        # Get Federal Funds Rate for monetary policy context
        # Fed Funds Rate helps assess the policy stance and potential duration risk
        # High Fed Funds Rate suggests restrictive policy, affecting bond pricing
        fed_funds = get_fred_data("FEDFUNDS", start_date, end_date, verbose_data)
        if fed_funds is not None and len(fed_funds) > 0:
            yield_data["fed_funds"] = fed_funds.iloc[-1]["value"]
        else:
            logger.warning("Failed to fetch Federal Funds Rate data from FRED", module="fixed_income_agent")
        
    except Exception as e:
        import traceback
        logger.error(f"Error fetching yield curve data: {e}", module="fixed_income_agent")
        logger.error(f"Traceback: {traceback.format_exc()}", module="fixed_income_agent")
        # Set default values if data fetch fails
        yield_data = {
            "treasury_2y": 4.5,
            "treasury_10y": 4.2,
            "yield_spread": -0.3,
            "curve_shape": "inverted",
            "fed_funds": 5.0
        }
    
    return yield_data


def make_bond_decisions(
    bond_allocation: float,
    yield_curve_data: dict,
    portfolio: dict,
    end_date: str,
    verbose_data: bool = False
) -> dict:
    """Make algorithmic bond investment decisions based on yield curve"""
    
    curve_shape = yield_curve_data.get("curve_shape", "normal")
    yield_spread = yield_curve_data.get("yield_spread", 0.0)
    
    # Algorithmic decision logic based on yield curve with finer granularity
    if curve_shape == "very_steep":
        # Very steep curve: maximum long-term bond allocation
        short_term_allocation = 0.2  # 20% SHY
        long_term_allocation = 0.8   # 80% TLT
        reasoning = f"Very steep yield curve (spread: {yield_spread:.2f}%) signals exceptional long-term bond opportunity"
        
    elif curve_shape == "steep":
        # Steep curve: favor long-term bonds (higher expected returns)
        short_term_allocation = 0.3  # 30% SHY
        long_term_allocation = 0.7   # 70% TLT
        reasoning = f"Steep yield curve (spread: {yield_spread:.2f}%) signals higher expected returns on long-term bonds"
        
    elif curve_shape == "normal":
        # Normal curve: balanced allocation
        short_term_allocation = 0.5  # 50% SHY
        long_term_allocation = 0.5   # 50% TLT
        reasoning = f"Normal yield curve (spread: {yield_spread:.2f}%) suggests balanced duration exposure"
        
    elif curve_shape == "flat":
        # Flat curve: slight preference for short-term
        short_term_allocation = 0.6  # 60% SHY
        long_term_allocation = 0.4   # 40% TLT
        reasoning = f"Flat yield curve (spread: {yield_spread:.2f}%) indicates economic uncertainty, favoring shorter duration"
        
    elif curve_shape == "inverted":
        # Inverted curve: strongly favor short-term bonds
        short_term_allocation = 0.8  # 80% SHY
        long_term_allocation = 0.2   # 20% TLT
        reasoning = f"Inverted yield curve (spread: {yield_spread:.2f}%) signals economic uncertainty and potential recession, strongly favoring short-term bonds"
        
    else:  # deeply_inverted
        # Deeply inverted curve: maximum short-term allocation
        short_term_allocation = 0.9  # 90% SHY
        long_term_allocation = 0.1   # 10% TLT
        reasoning = f"Deeply inverted yield curve (spread: {yield_spread:.2f}%) signals severe economic stress, maximum short-term bond allocation"
    
    # Calculate total portfolio value for bond allocation
    portfolio_value = portfolio.get("cash", 0) + sum(portfolio.get("cost_basis", {}).values())
    total_bond_capital = portfolio_value * bond_allocation
    
    # Calculate target values for each bond ETF
    shy_target_value = total_bond_capital * short_term_allocation
    tlt_target_value = total_bond_capital * long_term_allocation
    
    # Get current bond positions from portfolio
    positions = portfolio.get("positions", {})
    shy_current_shares = positions.get("SHY", {}).get("long", 0)
    tlt_current_shares = positions.get("TLT", {}).get("long", 0)
    
    # Get current prices for bond ETFs
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=1)
    start_date = start_dt.strftime("%Y-%m-%d")
    
    # Fetch SHY price
    shy_prices = get_prices("SHY", start_date, end_date, verbose_data)
    shy_prices_df = prices_to_df(shy_prices)
    shy_current_price = shy_prices_df["close"].iloc[-1]
    
    # Fetch TLT price
    tlt_prices = get_prices("TLT", start_date, end_date, verbose_data)
    tlt_prices_df = prices_to_df(tlt_prices)
    tlt_current_price = tlt_prices_df["close"].iloc[-1]
    
    # Calculate current position values
    shy_current_value = shy_current_shares * shy_current_price
    tlt_current_value = tlt_current_shares * tlt_current_price
    
    # Calculate target shares based on target values
    shy_target_shares = int(shy_target_value / shy_current_price) if shy_current_price > 0 else 0
    tlt_target_shares = int(tlt_target_value / tlt_current_price) if tlt_current_price > 0 else 0
    
    # Calculate share differences (positive = buy, negative = sell)
    shy_share_diff = shy_target_shares - shy_current_shares
    tlt_share_diff = tlt_target_shares - tlt_current_shares
    
    # Determine actions and quantities for SHY
    if shy_share_diff > 0:
        shy_action = "buy"
        shy_quantity = shy_share_diff
    elif shy_share_diff < 0:
        shy_action = "sell"
        shy_quantity = abs(shy_share_diff)
    else:
        shy_action = "hold"
        shy_quantity = 0
    
    # Determine actions and quantities for TLT
    if tlt_share_diff > 0:
        tlt_action = "buy"
        tlt_quantity = tlt_share_diff
    elif tlt_share_diff < 0:
        tlt_action = "sell"
        tlt_quantity = abs(tlt_share_diff)
    else:
        tlt_action = "hold"
        tlt_quantity = 0
    
    # Create decisions in portfolio manager format
    decisions = {
        "SHY": {
            "action": shy_action,
            "quantity": shy_quantity,
            "confidence": 85.0,  # High confidence for algorithmic bond allocation
            "reasoning": f"Short-term Treasury allocation: {reasoning}. Target: {shy_target_shares} shares, Current: {shy_current_shares} shares"
        },
        "TLT": {
            "action": tlt_action,
            "quantity": tlt_quantity,
            "confidence": 85.0,  # High confidence for algorithmic bond allocation
            "reasoning": f"Long-term Treasury allocation: {reasoning}. Target: {tlt_target_shares} shares, Current: {tlt_current_shares} shares"
        },
        "bond_analysis": {  # Analysis data that won't be processed as ticker decisions
            "bond_allocation_percentage": bond_allocation * 100,
            "total_bond_capital": total_bond_capital,
            "current_bond_value": shy_current_value + tlt_current_value,
            "target_bond_value": shy_target_value + tlt_target_value,
            "yield_curve_analysis": {
                "curve_shape": curve_shape,
                "yield_spread": yield_spread,
                "treasury_2y": yield_curve_data.get("treasury_2y", 0),
                "treasury_10y": yield_curve_data.get("treasury_10y", 0),
                "reasoning": reasoning
            }
        }
    }
    
    if verbose_data:
        logger.debug(f"Algorithmic bond decisions:", module="fixed_income_agent")
        logger.debug(f"  Curve shape: {curve_shape}", module="fixed_income_agent")
        logger.debug(f"  SHY allocation: {short_term_allocation*100:.1f}%", module="fixed_income_agent")
        logger.debug(f"  TLT allocation: {long_term_allocation*100:.1f}%", module="fixed_income_agent")
        logger.debug(f"  Total bond capital: ${total_bond_capital:,.2f}", module="fixed_income_agent")
    
    return decisions