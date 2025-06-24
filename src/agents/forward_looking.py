from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from tools.fred_api import get_vix_data
import json
from utils.logger import logger


def forward_looking_agent(state: AgentState):
    """
    Determines asset allocation between stocks and bonds based on VIX (market volatility expectations).
    Based on research proposal: adjusts allocation based on implied market volatility.
    
    Logic:
    - Low VIX (low expected volatility) → favor stocks (stable market conditions)
    - High VIX (high expected volatility) → favor bonds (market uncertainty, flight to safety)
    """
    
    # Get verbose_data from metadata or default to False
    verbose_data = state["metadata"].get("verbose_data", False)
    logger.debug("Accessing Forward-Looking Agent", module="forward_looking_agent")
    
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    
    progress.update_status("forward_looking_agent", "VIX", "Fetching VIX volatility data")
    
    # Get VIX data
    vix_data = get_vix_data(start_date, end_date, verbose_data)
    
    # Debug: Log what VIX data we actually have
    if verbose_data:
        logger.debug(f"=== VIX DATA ANALYSIS for {end_date} ===", module="forward_looking_agent")
        if vix_data is not None and not vix_data.empty:
            logger.debug(f"VIX data:", module="forward_looking_agent")
            logger.debug(f"  - Shape: {vix_data.shape}", module="forward_looking_agent")
            logger.debug(f"  - Date range: {vix_data.index.min()} to {vix_data.index.max()}", module="forward_looking_agent")
            logger.debug(f"  - Current VIX: {vix_data['value'].iloc[-1]:.2f}", module="forward_looking_agent")
        else:
            logger.debug(f"VIX data: EMPTY OR NONE", module="forward_looking_agent")
    
    if vix_data is None or vix_data.empty:
        progress.update_status("forward_looking_agent", "VIX", "Failed: No VIX data found")
        logger.warning("No VIX data available for allocation decision", module="forward_looking_agent")
        allocation = {"stock_allocation": 0.6, "bond_allocation": 0.4, "reasoning": "No VIX data available, using neutral allocation"}
    else:
        progress.update_status("forward_looking_agent", "VIX", "Analyzing volatility indicators")
        allocation = analyze_vix_level(vix_data, verbose_data)
    
    progress.update_status("forward_looking_agent", "VIX", "Done")
    
    message = HumanMessage(
        content=json.dumps(allocation),
        name="forward_looking_agent",
    )
    
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(allocation, "Forward-Looking Agent")
    
    # Add the allocation to the analyst_signals
    state["data"]["analyst_signals"]["forward_looking_agent"] = allocation
    
    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def analyze_vix_level(vix_data, verbose_data=False):
    """
    Analyze current VIX level to determine stock vs bond allocation.
    
    Args:
        vix_data: DataFrame with VIX values
        verbose_data: Optional flag for verbose logging
    
    Returns:
        Dictionary with stock_allocation, bond_allocation, and reasoning
    """
    
    if vix_data is None or vix_data.empty:
        return {
            "stock_allocation": 0.6,
            "bond_allocation": 0.4,
            "reasoning": "No VIX data available, using neutral allocation"
        }
    
    # Get current VIX level
    current_vix = vix_data['value'].iloc[-1]
    
    # VIX interpretation thresholds based on typical market conditions
    if current_vix < 15:  # Low volatility - market complacency
        stock_allocation = 0.7
        bond_allocation = 0.3
        reasoning = f"Low VIX ({current_vix:.1f}) indicates market stability and low expected volatility, favoring higher stock allocation (70/30)"
    elif current_vix > 30:  # High volatility - market fear
        stock_allocation = 0.4
        bond_allocation = 0.6
        reasoning = f"High VIX ({current_vix:.1f}) indicates market fear and high expected volatility, favoring higher bond allocation (40/60)"
    elif current_vix > 20:  # Elevated volatility
        stock_allocation = 0.5
        bond_allocation = 0.5
        reasoning = f"Elevated VIX ({current_vix:.1f}) indicates increased market uncertainty, using balanced allocation (50/50)"
    else:  # Normal volatility range (15-20)
        stock_allocation = 0.6
        bond_allocation = 0.4
        reasoning = f"Normal VIX ({current_vix:.1f}) indicates moderate market volatility, maintaining standard allocation (60/40)"
    
    return {
        "stock_allocation": round(stock_allocation, 2),
        "bond_allocation": round(bond_allocation, 2),
        "current_vix": round(current_vix, 2),
        "reasoning": reasoning
    }