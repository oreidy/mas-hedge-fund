import json
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
from utils.logger import logger



class EquityDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: float = Field(description="Confidence in the decision, between 0.0 and 100.0")
    reasoning: str = Field(description="Reasoning for the decision")


class EquityAgentOutput(BaseModel):
    decisions: dict[str, EquityDecision] = Field(description="Dictionary of ticker to trading decisions")


##### Equity Agent #####
def equity_agent(state: AgentState):
    """Makes final trading decisions and generates orders for multiple tickers"""

    # Get verbose_data from metadata or default to False
    verbose_data = state["metadata"].get("verbose_data", False)
    logger.debug("Accessing Equity Agent", module="equity_agent")

    # Get the portfolio and analyst signals
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    progress.update_status("equity_agent", None, "Analyzing signals")

    # Get position limits, current prices, and signals for every ticker
    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status("equity_agent", ticker, "Processing analyst signals")

        # Get position limits and current prices for the ticker
        # Asset allocation (stocks vs bonds) is handled upstream by the risk manager.
        # This equity agent focuses purely on individual stock selection and
        # position sizing within the allocated stock capital.
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0)
        current_prices[ticker] = risk_data.get("current_price", 0)

        # Calculate maximum shares allowed based on position limit and price
        if current_prices[ticker] > 0 and not pd.isna(current_prices[ticker]) and not pd.isna(position_limits[ticker]):
            max_shares[ticker] = int(position_limits[ticker] / current_prices[ticker])
        else:
            max_shares[ticker] = 0

        # Get signals for the ticker
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if agent != "risk_management_agent" and ticker in signals:
                ticker_signals[agent] = {"signal": signals[ticker]["signal"], "confidence": signals[ticker]["confidence"]}
        signals_by_ticker[ticker] = ticker_signals

    progress.update_status("equity_agent", None, "Making trading decisions")

    # Generate the trading decision with batching to handle token limits
    result = generate_trading_decision_batched(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        model_name=state["metadata"]["model_name"],
        model_provider=state["metadata"]["model_provider"],
        batch_size=15  # Process 15 tickers at a time
    )

    # Create the equity agent message
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name="equity_agent",
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}, "Equity Agent")

    progress.update_status("equity_agent", None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float],
    model_name: str,
    model_provider: str,
) -> EquityAgentOutput:
    """Attempts to get a decision from the LLM with retry logic"""
    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
              "system",
              """You are an equity portfolio manager agent making final trading decisions based on multiple tickers.

              Trading Rules:
              - For long positions:
                * Only buy if you have available cash
                * Only sell if you currently hold long shares of that ticker
                * Sell quantity must be ≤ current long position shares
                * Buy quantity must be ≤ max_shares for that ticker
              
              - For short positions:
                * Only short if you have available margin (50% of position value required)
                * Only cover if you currently have short shares of that ticker
                * Cover quantity must be ≤ current short position shares
                * Short quantity must respect margin requirements
              
              - The max_shares values are pre-calculated to respect position limits
              - Consider both long and short opportunities based on signals
              - Maintain appropriate risk management with both long and short exposure

              Available Actions (use EXACTLY these words):
              - "buy": Open or add to long position
              - "sell": Close or reduce long position
              - "short": Open or add to short position
              - "cover": Close or reduce short position
              - "hold": No action
              
              IMPORTANT: Use only these 5 exact action words: buy, sell, short, cover, hold

              Inputs:
              - signals_by_ticker: dictionary of ticker → signals
              - max_shares: maximum shares allowed per ticker
              - portfolio_cash: current cash in portfolio
              - portfolio_positions: current positions (both long and short)
              - current_prices: current prices for each ticker
              - margin_requirement: current margin requirement for short positions
              """,
            ),
            (
              "human",
              """Based on the team's analysis, make your trading decisions for each ticker.

              Here are the signals by ticker:
              {signals_by_ticker}

              Current Prices:
              {current_prices}

              Maximum Shares Allowed For Purchases:
              {max_shares}

              Portfolio Cash: {portfolio_cash}
              Current Positions: {portfolio_positions}
              Current Margin Requirement: {margin_requirement}

              Output strictly in JSON with the following structure:
              {{
                "decisions": {{
                  "TICKER1": {{
                    "action": "buy/sell/short/cover/hold",
                    "quantity": integer,
                    "confidence": float,
                    "reasoning": "string"
                  }},
                  "TICKER2": {{
                    ...
                  }},
                  ...
                }}
              }}
              """,
            ),
        ]
    )

    
    # Generate the prompt
    prompt = template.invoke(
        {
            "signals_by_ticker": json.dumps(signals_by_ticker, indent=2),
            "current_prices": json.dumps(current_prices, indent=2),
            "max_shares": json.dumps(max_shares, indent=2),
            "portfolio_cash": f"{portfolio.get('cash', 0):.2f}",
            "portfolio_positions": json.dumps(portfolio.get('positions', {}), indent=2),
            "margin_requirement": f"{portfolio.get('margin_requirement', 0):.2f}",
        }
    )

    # Create default factory for EquityAgentOutput
    def create_default_equity_output():
        return EquityAgentOutput(decisions={ticker: EquityDecision(action="hold", quantity=0, confidence=0.0, reasoning="Error in equity agent, defaulting to hold") for ticker in tickers})

    return call_llm(prompt=prompt, model_name=model_name, model_provider=model_provider, pydantic_model=EquityAgentOutput, agent_name="equity_agent", default_factory=create_default_equity_output)


def generate_trading_decision_batched(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float],
    model_name: str,
    model_provider: str,
    batch_size: int = 50,
) -> EquityAgentOutput:
    """Process tickers in batches to handle token limits"""
    
    all_decisions = {}
    
    # Process tickers in batches
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]
        
        # Filter data for this batch
        batch_signals = {ticker: signals_by_ticker[ticker] for ticker in batch_tickers}
        batch_prices = {ticker: current_prices[ticker] for ticker in batch_tickers}
        batch_max_shares = {ticker: max_shares[ticker] for ticker in batch_tickers}
        
        # Filter portfolio positions to only include batch tickers to reduce token usage
        all_positions = portfolio.get('positions', {})
        batch_positions = {
            ticker: all_positions.get(ticker, {
                'long': 0, 'short': 0, 'long_cost_basis': 0.0, 'short_cost_basis': 0.0
            }) for ticker in batch_tickers
        }
        batch_portfolio = {
            'cash': portfolio.get('cash', 0),
            'margin_requirement': portfolio.get('margin_requirement', 0),
            'positions': batch_positions
        }
        
        # Process this batch
        batch_result = generate_trading_decision(
            tickers=batch_tickers,
            signals_by_ticker=batch_signals,
            current_prices=batch_prices,
            max_shares=batch_max_shares,
            portfolio=batch_portfolio,
            model_name=model_name,
            model_provider=model_provider,
        )
        
        # Merge decisions
        all_decisions.update(batch_result.decisions)
    
    return EquityAgentOutput(decisions=all_decisions)