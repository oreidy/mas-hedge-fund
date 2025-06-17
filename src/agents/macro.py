from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from tools.fred_api import get_macro_indicators_for_allocation
import json
from utils.logger import logger


def macro_agent(state: AgentState):
    """
    Determines asset allocation between stocks and bonds based on macroeconomic indicators.
    Based on research proposal: uses CPI, PCE, GDP growth, and Federal Funds Rate.
    
    Logic:
    - High inflation (CPI, PCE) and high interest rates (Fed Funds) → favor bonds
    - Strong economic growth (GDP) → favor stocks
    - Weak growth → favor bonds (safe-haven)
    """
    
    # Get verbose_data from metadata or default to False
    verbose_data = state["metadata"].get("verbose_data", False)
    logger.debug("Accessing Macro Agent", module="macro_agent")
    
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    
    progress.update_status("macro_agent", "MACRO", "Fetching macroeconomic indicators")
    
    # Get macro indicators (excluding VIX for now)
    macro_data = get_macro_indicators_for_allocation(start_date, end_date, verbose_data)
    
    # Remove VIX from analysis for pure macro agent
    if 'vix' in macro_data:
        del macro_data['vix']
    
    if not any(not df.empty for df in macro_data.values()):
        progress.update_status("macro_agent", "MACRO", "Failed: No macro data found")
        logger.warning("No macro data available for allocation decision", module="macro_agent")
        allocation = {"stock_allocation": 0.5, "bond_allocation": 0.5, "reasoning": "No data available, using neutral allocation"}
    else:
        progress.update_status("macro_agent", "MACRO", "Analyzing macro indicators")
        allocation = analyze_macro_indicators(macro_data, verbose_data)
    
    progress.update_status("macro_agent", "MACRO", "Done")
    
    message = HumanMessage(
        content=json.dumps(allocation),
        name="macro_agent",
    )
    
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(allocation, "Macro Agent")
    
    # Add the allocation to the analyst_signals
    state["data"]["analyst_signals"]["macro_agent"] = allocation
    
    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


def analyze_macro_indicators(macro_data, verbose_data=False):
    """
    Analyze macro indicators to determine stock vs bond allocation.
    
    Returns:
        Dictionary with stock_allocation, bond_allocation, and reasoning
    """
    
    signals = []
    reasoning = {}
    
    # 1. Inflation Analysis (CPI and PCE)
    inflation_signal = analyze_inflation(macro_data, verbose_data)
    signals.append(inflation_signal["signal"])
    reasoning["inflation"] = inflation_signal
    
    # 2. Interest Rate Analysis (Federal Funds Rate)
    interest_rate_signal = analyze_interest_rates(macro_data, verbose_data)
    signals.append(interest_rate_signal["signal"])
    reasoning["interest_rates"] = interest_rate_signal
    
    # 3. Economic Growth Analysis (GDP)
    growth_signal = analyze_economic_growth(macro_data, verbose_data)
    signals.append(growth_signal["signal"])
    reasoning["economic_growth"] = growth_signal
    
    # Determine overall allocation
    bond_signals = signals.count("bonds")
    stock_signals = signals.count("stocks")
    neutral_signals = signals.count("neutral")
    
    # Base allocation: 60% stocks, 40% bonds (typical balanced portfolio)
    base_stock_allocation = 0.6
    base_bond_allocation = 0.4
    
    # Adjust based on signals (each signal can shift allocation by 10%)
    net_signal_strength = stock_signals - bond_signals
    allocation_adjustment = net_signal_strength * 0.1
    
    stock_allocation = max(0.2, min(0.8, base_stock_allocation + allocation_adjustment))
    bond_allocation = 1.0 - stock_allocation
    
    return {
        "stock_allocation": round(stock_allocation, 2),
        "bond_allocation": round(bond_allocation, 2),
        "signal_summary": {
            "stock_signals": stock_signals,
            "bond_signals": bond_signals,
            "neutral_signals": neutral_signals
        },
        "reasoning": reasoning
    }


def analyze_inflation(macro_data, verbose_data=False):
    """Analyze inflation indicators (CPI and PCE)"""
    
    cpi_data = macro_data.get('cpi', None)
    pce_data = macro_data.get('pce', None)
    
    if (cpi_data is None or cpi_data.empty) and (pce_data is None or pce_data.empty):
        return {"signal": "neutral", "details": "No inflation data available"}
    
    # Calculate recent inflation trends (last 12 months if available)
    inflation_signals = []
    details = []
    
    # CPI Analysis
    if cpi_data is not None and not cpi_data.empty and len(cpi_data) >= 12:
        recent_cpi = cpi_data.tail(12)
        cpi_growth = ((recent_cpi['value'].iloc[-1] / recent_cpi['value'].iloc[0]) - 1) * 100
        
        if cpi_growth > 3.0:  # High inflation threshold
            inflation_signals.append("bonds")
            details.append(f"High CPI inflation: {cpi_growth:.1f}%")
        elif cpi_growth < 1.0:  # Low inflation
            inflation_signals.append("stocks")
            details.append(f"Low CPI inflation: {cpi_growth:.1f}%")
        else:
            inflation_signals.append("neutral")
            details.append(f"Moderate CPI inflation: {cpi_growth:.1f}%")
    
    # PCE Analysis
    if pce_data is not None and not pce_data.empty and len(pce_data) >= 12:
        recent_pce = pce_data.tail(12)
        pce_growth = ((recent_pce['value'].iloc[-1] / recent_pce['value'].iloc[0]) - 1) * 100
        
        if pce_growth > 2.5:  # High inflation threshold (PCE target is 2%)
            inflation_signals.append("bonds")
            details.append(f"High PCE inflation: {pce_growth:.1f}%")
        elif pce_growth < 1.5:  # Low inflation
            inflation_signals.append("stocks")
            details.append(f"Low PCE inflation: {pce_growth:.1f}%")
        else:
            inflation_signals.append("neutral")
            details.append(f"Moderate PCE inflation: {pce_growth:.1f}%")
    
    # Determine overall inflation signal
    if not inflation_signals:
        overall_signal = "neutral"
    elif inflation_signals.count("bonds") > inflation_signals.count("stocks"):
        overall_signal = "bonds"
    elif inflation_signals.count("stocks") > inflation_signals.count("bonds"):
        overall_signal = "stocks"
    else:
        overall_signal = "neutral"
    
    return {
        "signal": overall_signal,
        "details": "; ".join(details) if details else "Insufficient data for inflation analysis"
    }


def analyze_interest_rates(macro_data, verbose_data=False):
    """Analyze Federal Funds Rate"""
    
    fed_funds_data = macro_data.get('fed_funds', None)
    
    if fed_funds_data is None or fed_funds_data.empty:
        return {"signal": "neutral", "details": "No Fed Funds Rate data available"}
    
    # Get current Fed Funds Rate
    current_rate = fed_funds_data['value'].iloc[-1]
    
    # Analyze rate level and trend
    if current_rate > 4.0:  # High interest rate environment
        signal = "bonds"
        details = f"High Fed Funds Rate: {current_rate:.2f}% favors bonds"
    elif current_rate < 2.0:  # Low interest rate environment
        signal = "stocks"
        details = f"Low Fed Funds Rate: {current_rate:.2f}% favors stocks"
    else:
        signal = "neutral"
        details = f"Moderate Fed Funds Rate: {current_rate:.2f}%"
    
    # Check rate trend if we have enough data
    if len(fed_funds_data) >= 6:
        rate_6m_ago = fed_funds_data['value'].iloc[-6]
        rate_change = current_rate - rate_6m_ago
        
        if rate_change > 0.5:  # Rising rates
            signal = "bonds" if signal != "stocks" else "neutral"
            details += f"; Rising trend (+{rate_change:.2f}%)"
        elif rate_change < -0.5:  # Falling rates
            signal = "stocks" if signal != "bonds" else "neutral"
            details += f"; Falling trend ({rate_change:.2f}%)"
    
    return {"signal": signal, "details": details}


def analyze_economic_growth(macro_data, verbose_data=False):
    """Analyze GDP growth"""
    
    gdp_data = macro_data.get('gdp', None)
    
    if gdp_data is None or gdp_data.empty or len(gdp_data) < 4:
        return {"signal": "neutral", "details": "Insufficient GDP data available"}
    
    # Calculate GDP growth rate (quarterly)
    recent_gdp = gdp_data.tail(4)  # Last 4 quarters
    if len(recent_gdp) >= 2:
        gdp_growth = ((recent_gdp['value'].iloc[-1] / recent_gdp['value'].iloc[-2]) - 1) * 100
        
        if gdp_growth > 2.5:  # Strong growth
            signal = "stocks"
            details = f"Strong GDP growth: {gdp_growth:.1f}% favors stocks"
        elif gdp_growth < 1.0:  # Weak growth
            signal = "bonds"
            details = f"Weak GDP growth: {gdp_growth:.1f}% favors bonds"
        else:
            signal = "neutral"
            details = f"Moderate GDP growth: {gdp_growth:.1f}%"
    else:
        signal = "neutral"
        details = "Insufficient GDP data for growth calculation"
    
    return {"signal": signal, "details": details}