from graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from tools.api import get_financial_metrics, get_market_cap, search_line_items, get_outstanding_shares
from utils.llm import call_llm
from utils.progress import progress
from utils.logger import logger


class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


class WarrenBuffettBatchOutput(BaseModel):
    decisions: dict[str, WarrenBuffettSignal]


def warren_buffett_agent(state: AgentState):
    """Analyzes stocks using Buffett's principles and LLM reasoning."""

    # Get verbose_data from metadata or default to False
    verbose_data = state["metadata"].get("verbose_data", False)
    logger.debug("Accessing Warren Buffet Agent", module="warren_buffet_agent")

    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Phase 1: Collect all analysis data for all tickers (no LLM calls yet)
    analysis_data = {}

    for ticker in tickers:
        progress.update_status("warren_buffett_agent", ticker, "Fetching financial metrics")
        # Fetch required data
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5, verbose_data=verbose_data)

        progress.update_status("warren_buffett_agent", ticker, "Gathering financial line items")
        try:
            financial_line_items = search_line_items(
                ticker,
                [
                    "capital_expenditure",
                    "depreciation_and_amortization",
                    "net_income",
                    "outstanding_shares",
                    "total_assets",
                    "total_liabilities",
                ],
                end_date,
                period="annual",
                limit=5,
                verbose_data=verbose_data
            )
        except Exception as e:
            logger.warning(f"Unable to gather required financial line items for {ticker}. This may be due to company type (e.g., banks have different financial structures): {e}", module="warren_buffett_agent")
            # Skip this ticker and continue with neutral signal
            buffett_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": f"Unable to analyze {ticker} using Warren Buffett methodology - insufficient or incompatible financial data"
            }
            continue

        # Get current market cap
        progress.update_status("warren_buffett_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, verbose_data)

        # Analyze fundamentals
        progress.update_status("warren_buffett_agent", ticker, "Analyzing fundamentals")
        fundamental_analysis = analyze_fundamentals(metrics, verbose_data)

        # Analyze consistency
        progress.update_status("warren_buffett_agent", ticker, "Analyzing consistency")
        consistency_analysis = analyze_consistency(financial_line_items, verbose_data)

        #Calculating intrinsic value
        progress.update_status("warren_buffett_agent", ticker, "Calculating intrinsic value")
        intrinsic_value_analysis = calculate_intrinsic_value(financial_line_items, verbose_data)

        # Calculate total score
        total_score = fundamental_analysis["score"] + consistency_analysis["score"]
        max_possible_score = 10

        # Add margin of safety analysis if we have both intrinsic value and current price
        margin_of_safety = None
        intrinsic_value = intrinsic_value_analysis["intrinsic_value"]
        if intrinsic_value and market_cap:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap

            # Add to score if there's a good margin of safety (>30%)
            if margin_of_safety > 0.3:
                total_score += 2
                max_possible_score += 2

        # Generate trading signal
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        # Combine all analysis results
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "margin_of_safety": margin_of_safety or 0.0,
            "fundamental_score": fundamental_analysis["score"],
            "consistency_score": consistency_analysis["score"],
            "fundamental_summary": fundamental_analysis["details"],  
            "consistency_summary": consistency_analysis["details"],  
        }

        logger.debug(f"===Analysis results: signal={signal}, score={total_score}/{max_possible_score}===", module="warren_buffet_agent", ticker=ticker)
        logger.debug(f"- Fundamentals score: {fundamental_analysis['score']}", module="warren_buffet_agent", ticker=ticker)
        logger.debug(f"- Fundamentals details: {fundamental_analysis['details']}", module="warren_buffet_agent", ticker=ticker)
        logger.debug(f"- Consistency score: {consistency_analysis['score']}", module="warren_buffet_agent", ticker=ticker)
        logger.debug(f"- Consistency details: {consistency_analysis['details']}", module="warren_buffet_agent", ticker=ticker) 
        logger.debug(f"- Instrinsic value: {intrinsic_value_analysis['intrinsic_value']}; Owner Earnigns: {intrinsic_value_analysis['owner_earnings']}", module="warren_buffet_agent", ticker=ticker)
        logger.debug(f"- Instrinsic value details: {intrinsic_value_analysis['details']}", module="warren_buffet_agent", ticker=ticker)

        progress.update_status("warren_buffett_agent", ticker, "Analysis complete")
    
    # Phase 2: Process all collected data through batched LLM calls
    progress.update_status("warren_buffett_agent", None, "Generating batched Buffett analysis")
    
    buffett_decisions = generate_buffett_output_batched(
        analysis_data_by_ticker=analysis_data,
        model_name=state["metadata"]["model_name"],
        model_provider=state["metadata"]["model_provider"],
        batch_size=30
    )
    
    # Convert decisions to the expected format
    buffett_analysis = {}
    for ticker, decision in buffett_decisions.items():
        buffett_analysis[ticker] = {
            "signal": decision.signal,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning
        }
        progress.update_status("warren_buffett_agent", ticker, "Done")

    # Create the message
    message = HumanMessage(content=json.dumps(buffett_analysis), name="warren_buffett_agent")

    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(buffett_analysis, "Warren Buffett Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["warren_buffett_agent"] = buffett_analysis

    return {"messages": [message], "data": state["data"]}


def generate_buffett_output_batched(
    analysis_data_by_ticker: dict[str, dict],
    model_name: str,
    model_provider: str,
    batch_size: int = 30,
) -> dict[str, WarrenBuffettSignal]:
    """Process tickers in batches to handle token limits"""
    
    all_decisions = {}
    tickers = list(analysis_data_by_ticker.keys())
    
    # Process tickers in batches
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]
        
        # Filter analysis data for this batch
        batch_analysis_data = {ticker: analysis_data_by_ticker[ticker] for ticker in batch_tickers}
        
        # Process this batch
        batch_result = generate_buffett_output_batch(
            analysis_data_by_ticker=batch_analysis_data,
            model_name=model_name,
            model_provider=model_provider,
        )
        
        # Merge decisions
        all_decisions.update(batch_result.decisions)
    
    return all_decisions


def generate_buffett_output_batch(
    analysis_data_by_ticker: dict[str, dict],
    model_name: str,
    model_provider: str,
) -> WarrenBuffettBatchOutput:
    """Generate investment decisions for multiple tickers in a single LLM call"""
    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
            "system",
            """You are Warren Buffett making investment decisions. Focus on: strong operating margins (moats), consistent earnings, conservative debt, high ROE, and margin of safety (>30%). Provide rational value-based recommendations for multiple companies."""
            ),
            (
            "human",
            """Analyze the following companies and provide investment decisions for each:

            {ticker_analyses}

            Return JSON in this exact format:
            {{
                "decisions": {{
                    "TICKER1": {{"signal": "bullish/bearish/neutral", "confidence": float (0-100), "reasoning": "brief explanation"}},
                    "TICKER2": {{"signal": "bullish/bearish/neutral", "confidence": float (0-100), "reasoning": "brief explanation"}},
                    ...
                }}
            }}
            """
            )
        ]   
    )

    # Format ticker analyses for the prompt
    ticker_analyses_text = []
    for ticker, data in analysis_data_by_ticker.items():
        margin_safety = data["margin_of_safety"]
        margin_text = f"{margin_safety:.1%}" if margin_safety else "Not calculated"
        analysis_text = f"""{ticker}:
- Total Score: {data["score"]}/{data.get("max_score", 12)}
- Margin of Safety: {margin_text}
- Fundamental Score: {data["fundamental_score"]}/7
- Consistency Score: {data["consistency_score"]}/3
- Fundamental Details: {data["fundamental_summary"]}
- Consistency Details: {data["consistency_summary"]}"""
        ticker_analyses_text.append(analysis_text)
    
    prompt = template.invoke({
        "ticker_analyses": "\n\n".join(ticker_analyses_text)
    })

    def create_default_warren_buffett_batch_output():
        default_decisions = {}
        for ticker in analysis_data_by_ticker.keys():
            default_decisions[ticker] = WarrenBuffettSignal(
                signal="neutral",
                confidence=0.0,
                reasoning="Error in analysis, defaulting to neutral"
            )
        return WarrenBuffettBatchOutput(decisions=default_decisions)

    return call_llm(
        prompt=prompt, 
        model_name=model_name, 
        model_provider=model_provider, 
        pydantic_model=WarrenBuffettBatchOutput, 
        agent_name="warren_buffett_agent", 
        default_factory=create_default_warren_buffett_batch_output,
    )


def analyze_fundamentals(metrics: list, verbose_data: bool = False) -> dict[str, any]:
    """Analyze company fundamentals based on Buffett's criteria."""

    if not metrics:
        return {"score": 0, "details": "Insufficient fundamental data"}

    # Get latest metrics
    latest_metrics = metrics[0]

    score = 0
    reasoning = []

    # Check ROE (Return on Equity)
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:  # 15% ROE threshold
        score += 2
        reasoning.append(f"Strong ROE of {latest_metrics.return_on_equity:.1%}")
    elif latest_metrics.return_on_equity:
        reasoning.append(f"Weak ROE of {latest_metrics.return_on_equity:.1%}")
    else:
        reasoning.append("ROE data not available")

    # Check Debt to Equity
    if latest_metrics.debt_to_equity and latest_metrics.debt_to_equity < 0.5:
        score += 2
        reasoning.append("Conservative debt levels")
    elif latest_metrics.debt_to_equity:
        reasoning.append(f"High debt to equity ratio of {latest_metrics.debt_to_equity:.1f}")
    else:
        reasoning.append("Debt to equity data not available")

    # Check Operating Margin
    if latest_metrics.operating_margin and latest_metrics.operating_margin > 0.15:
        score += 2
        reasoning.append("Strong operating margins")
    elif latest_metrics.operating_margin:
        reasoning.append(f"Weak operating margin of {latest_metrics.operating_margin:.1%}")
    else:
        reasoning.append("Operating margin data not available")

    # Check Current Ratio
    if latest_metrics.current_ratio and latest_metrics.current_ratio > 1.5:
        score += 1
        reasoning.append("Good liquidity position")
    elif latest_metrics.current_ratio:
        reasoning.append(f"Weak liquidity with current ratio of {latest_metrics.current_ratio:.1f}")
    else:
        reasoning.append("Current ratio data not available")
        
    if verbose_data:
        logger.debug(f"Final score: {score}", module="analyze_fundamentals")
        logger.debug(f"Final reasoning: {'; '.join(reasoning)}", module="analyze_fundamentals")

    return {"score": score, "details": "; ".join(reasoning)}


def analyze_consistency(financial_line_items: list, verbose_data: bool = False) -> dict[str, any]:
    """Analyze earnings consistency and growth."""
    if len(financial_line_items) < 4:  # Need at least 4 periods for trend analysis
        logger.debug("Insufficient historical data for consistency analysis", module="analyze_consistency")
        return {"score": 0, "details": "Insufficient historical data"}

    score = 0
    reasoning = []

    if verbose_data:
        for i, item in enumerate(financial_line_items):
            logger.debug(f"Period {i+1}: {item.report_period}, Net income: {item.net_income}", module="analyze_consistency")

    # Check earnings growth trend
    earnings_values = [item.net_income for item in financial_line_items if item.net_income]
    if len(earnings_values) >= 4:
        earnings_growth = all(earnings_values[i] > earnings_values[i + 1] for i in range(len(earnings_values) - 1))

        if earnings_growth:
            score += 3
            reasoning.append("Consistent earnings growth over past periods")
        else:
            reasoning.append("Inconsistent earnings growth pattern")

        # Calculate growth rate
        if len(earnings_values) >= 2:
            growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
            reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {len(earnings_values)} periods")
    else:
        reasoning.append("Insufficient earnings data for trend analysis")

    if verbose_data:
        logger.debug(f"Final score: {score}", module="analyze_consistency")
        logger.debug(f"Final reasoning: {'; '.join(reasoning)}", module="analyze_consistency")

    return {
        "score": score,
        "details": "; ".join(reasoning),
    }


def calculate_owner_earnings(financial_line_items: list, verbose_data: bool = False) -> dict[str, any]:
    """Calculate owner earnings (Buffett's preferred measure of true earnings power).
    Owner Earnings = Net Income + Depreciation - Maintenance CapEx"""
    if not financial_line_items or len(financial_line_items) < 1:
        logger.warning("calculate_owner_earnings: Insufficient data for owner earnings calculation", module="warren_buffett_agent")
        return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}

    latest = financial_line_items[0]

    # Get required components with error handling for different company types (e.g., banks)
    try:
        net_income = getattr(latest, 'net_income', None)
        depreciation = getattr(latest, 'depreciation_and_amortization', None)
        capex = getattr(latest, 'capital_expenditure', None)
    except AttributeError as e:
        logger.warning(f"calculate_owner_earnings: Missing financial line items for {latest.ticker}: {e}. This may be due to company type (e.g., banks have different financial structures)", module="warren_buffett_agent")
        return {"owner_earnings": None, "details": [f"Missing financial line items: {e}"]}

    if verbose_data:
        logger.debug(f"Net Income: {net_income}", module="calculate_owner_earnings")
        logger.debug(f"Depreciation: {depreciation}", module="calculate_owner_earnings")
        logger.debug(f"Capital Expenditure: {capex}", module="calculate_owner_earnings")


    if not all([net_income, depreciation, capex]):
        logger.warning("calculate_owner_earnings: Missing components for owner earnings calculation", module="warren_buffett_agent", ticker=latest.ticker)
        return {"owner_earnings": None, "details": ["Missing components for owner earnings calculation"]}

    # Estimate maintenance capex (typically 70-80% of total capex)
    maintenance_capex = capex * 0.75

    owner_earnings = net_income + depreciation - maintenance_capex

    if verbose_data:
        logger.debug(f"Calculated Owner Earnings: {owner_earnings}", module="calculate_owner_earnings")

    return {
        "owner_earnings": owner_earnings,
        "components": {"net_income": net_income, "depreciation": depreciation, "maintenance_capex": maintenance_capex},
        "details": ["Owner earnings calculated successfully"],
    }


def calculate_intrinsic_value(financial_line_items: list, verbose_data : bool = False) -> dict[str, any]:
    """Calculate intrinsic value using DCF with owner earnings."""

    if not financial_line_items:
        logger.warning("Insufficient financial line items for valuation", 
                       module="calculate_intrinsic_value")
        return {
            "intrinsic_value": None,
            "owner_earnings": None,
            "assumptions": {
                "growth_rate": 0.05,
                "discount_rate": 0.09,
                "terminal_multiple": 12,
                "projection_years": 10,
            },
            "details": ["Insufficient data for intrinsic value calculation"],
        }
        
    # Calculate owner earnings
    earnings_data = calculate_owner_earnings(financial_line_items)
    if not earnings_data["owner_earnings"]:
        ticker = financial_line_items[0].ticker if financial_line_items else "unknown"
        logger.warning("calculate_intrinsic_value: Owner earnings calculation failed", 
                       module="warren_buffett_agent", ticker=ticker)
        return {
            "intrinsic_value": None,
            "owner_earnings": None,
            "assumptions": {
                "growth_rate": 0.05,
                "discount_rate": 0.09,
                "terminal_multiple": 12,
                "projection_years": 10,
            },
            "details": ["Insufficient data for owner earnings calculation"],
        }

    owner_earnings = earnings_data["owner_earnings"]

    # Get current market data
    latest_financial_line_items = financial_line_items[0]
    ticker = latest_financial_line_items.ticker

    shares_outstanding = get_outstanding_shares(ticker, latest_financial_line_items.report_period, verbose_data=verbose_data)

    if not shares_outstanding:
        logger.warning(f"Missing outstanding_shares data for {ticker}. Using fallback valuation method.",
                      module="calculate_intrinsic_value", ticker=ticker)
        
        # Simplified estimation of business value as fallback
        intrinsic_value = owner_earnings * 15
        return {
            "intrinsic_value": intrinsic_value,
            "owner_earnings": owner_earnings,
            "assumptions": {
                "growth_rate": 0.05,
                "discount_rate": 0.09,
                "terminal_multiple": 15,  # Using a direct multiple for the fallback method
                "projection_years": 10,
            },
            "details": ["Intrinsic value estimated using simplified multiple method due to missing shares data"],
        }

    # Buffett's DCF assumptions
    growth_rate = 0.05  # Conservative 5% growth
    discount_rate = 0.09  # Typical 9% discount rate
    terminal_multiple = 12  # Conservative exit multiple
    projection_years = 10

    # Calculate future value
    future_value = 0
    for year in range(1, projection_years + 1):
        future_earnings = owner_earnings * (1 + growth_rate) ** year
        present_value = future_earnings / (1 + discount_rate) ** year
        future_value += present_value

    # Add terminal value
    terminal_value = (owner_earnings * (1 + growth_rate) ** projection_years * terminal_multiple) / (1 + discount_rate) ** projection_years
    intrinsic_value = future_value + terminal_value

    return {
        "intrinsic_value": intrinsic_value,
        "owner_earnings": owner_earnings,
        "assumptions": {
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_multiple": terminal_multiple,
            "projection_years": projection_years,
        },
        "details": ["Intrinsic value calculated using DCF model with owner earnings"],
    }


def generate_buffett_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> WarrenBuffettSignal:
    """Get investment decision from LLM with Buffett's principles"""
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Warren Buffett AI agent. Decide on investment signals based on Warren Buffett’s principles:

                strong operating margins (moats), consistent earnings, conservative debt, high ROE, and margin of safety (>30%). Provide rational value-based recommendations.
                """,
            ),
            (
                "human",
                """Analyze {ticker} as Warren Buffett would using Total Score={total_score}/10, Margin of Safety={margin_of_safety:.1%}.
                
                Fundamentals: {fundamental_details}
                Consistency: {consistency_details}

                Return JSON: {{"signal": "bullish/bearish/neutral", "confidence": float (0-100), "reasoning": "brief explanation"}}""",
            ),
        ]
    )

    ticker_data = analysis_data[ticker]
    prompt = template.invoke({
        "ticker": ticker,
        "total_score": ticker_data["score"],
        "margin_of_safety": ticker_data["margin_of_safety"] or 0.0,
        "fundamental_details": ticker_data["fundamental_summary"],
        "consistency_details": ticker_data["consistency_summary"]
    })

    # Create default factory for WarrenBuffettSignal
    def create_default_warren_buffett_signal():
        return WarrenBuffettSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt, 
        model_name=model_name, 
        model_provider=model_provider, 
        pydantic_model=WarrenBuffettSignal, 
        agent_name="warren_buffett_agent", 
        default_factory=create_default_warren_buffett_signal,
        )
