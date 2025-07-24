import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Back, Style, init
import questionary

from agents.bill_ackman import bill_ackman_agent
from agents.fundamentals import fundamentals_agent
from agents.equity_agent import equity_agent
from agents.technicals import technical_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.sentiment import sentiment_agent
from agents.warren_buffett import warren_buffett_agent
from agents.macro import macro_agent
from agents.forward_looking import forward_looking_agent
from agents.fixed_income import fixed_income_agent
from graph.state import AgentState
from agents.valuation import valuation_agent
from utils.display import print_trading_output
from utils.analysts import ANALYST_ORDER
from utils.progress import progress
from llm.models import LLM_ORDER, get_model_info
from utils.logger import logger, setup_logger, LogLevel
from utils.tickers import get_sp500_tickers

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tabulate import tabulate
from utils.visualize import save_graph_as_png
import time


# Load environment variables from .env file
load_dotenv()

init(autoreset=True)


def parse_hedge_fund_response(response):
    import json

    try:
        return json.loads(response)
    except:
        print(f"Error parsing response: {response}")
        return None


##### Run the Hedge Fund #####
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
    verbose_data: bool = False,
    open_positions: set = None,
):
    # Start progress tracking
    progress.start()

    try:
        # Create a new workflow if analysts are customized
        if selected_analysts:
            workflow = create_workflow(selected_analysts)
            agent = workflow.compile()
        else:
            agent = app

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data.",
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                    "open_positions": open_positions or set(),
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                    "verbose_data": verbose_data,
                },
            },
        )

        # Merge decisions from multiple agents (portfolio manager + fixed income)
        merged_decisions = {}
        
        # Look for messages from both equity agent and fixed income agents
        for message in final_state["messages"]:
            if hasattr(message, 'name'):
                if message.name == "equity_agent":
                    equity_decisions = parse_hedge_fund_response(message.content)
                    if equity_decisions:
                        merged_decisions.update(equity_decisions)
                elif message.name == "fixed_income_agent":
                    bond_decisions = parse_hedge_fund_response(message.content)
                    if bond_decisions:
                        # Only include SHY and TLT decisions, skip analysis data
                        for ticker, decision in bond_decisions.items():
                            if ticker in ["SHY", "TLT"] and isinstance(decision, dict) and "action" in decision:
                                merged_decisions[ticker] = decision

        return {
            "decisions": merged_decisions,
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        # Stop progress tracking
        progress.stop()


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Dictionary of all available analysts
    analyst_nodes = {
        "technical_analyst": ("technical_analyst_agent", technical_analyst_agent),
        "fundamentals_analyst": ("fundamentals_agent", fundamentals_agent),
        "sentiment_analyst": ("sentiment_agent", sentiment_agent),
        "valuation_analyst": ("valuation_agent", valuation_agent),
        "warren_buffett": ("warren_buffett_agent", warren_buffett_agent),
        "bill_ackman": ("bill_ackman_agent", bill_ackman_agent),
        "macro_analyst": ("macro_agent", macro_agent),
        "forward_looking_analyst": ("forward_looking_agent", forward_looking_agent),
    }

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    
    # Define stock analyst agents only
    stock_analyst_agents = {
        "technical_analyst",
        "fundamentals_analyst", 
        "sentiment_analyst",
        "valuation_analyst",
        "warren_buffett",
        "bill_ackman"
    }
    
    # Use optimized workflow if all stock analysts are selected (with or without macro/forward-looking)
    selected_set = set(selected_analysts)
    if stock_analyst_agents.issubset(selected_set):
        return create_optimized_workflow(selected_analysts)
    
    # For partial analyst selection, use original workflow
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and equity management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("equity_agent", equity_agent)

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    # Add fixed-income agent if macro OR forward-looking analysis is available
    include_fixed_income = "macro_analyst" in selected_analysts or "forward_looking_analyst" in selected_analysts
    
    if include_fixed_income:
        workflow.add_node("fixed_income_agent", fixed_income_agent)
        workflow.add_edge("risk_management_agent", "equity_agent")
        workflow.add_edge("risk_management_agent", "fixed_income_agent")
        workflow.add_edge("equity_agent", END)
        workflow.add_edge("fixed_income_agent", END)
    else:
        workflow.add_edge("risk_management_agent", "equity_agent")
        workflow.add_edge("equity_agent", END)

    workflow.set_entry_point("start_node")
    return workflow


def create_optimized_workflow(selected_analysts):
    """Create optimized workflow that filters tickers based on signal agreement."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)
    
    # Check which asset allocation agents are selected
    include_macro = "macro_analyst" in selected_analysts
    include_forward_looking = "forward_looking_analyst" in selected_analysts
    
    # First tier: Technical/Sentiment agents (run on all tickers) and conditionally add asset allocation agents
    workflow.add_node("technical_analyst_agent", technical_analyst_agent)
    workflow.add_node("sentiment_agent", sentiment_agent)
    workflow.add_edge("start_node", "technical_analyst_agent")
    workflow.add_edge("start_node", "sentiment_agent")
    
    # Conditionally add asset allocation agents
    if include_macro:
        workflow.add_node("macro_agent", macro_agent)
        workflow.add_edge("start_node", "macro_agent")
    
    if include_forward_looking:
        workflow.add_node("forward_looking_agent", forward_looking_agent)
        workflow.add_edge("start_node", "forward_looking_agent")
    
    # Ticker filtering node
    workflow.add_node("ticker_filter", ticker_filter_node)
    workflow.add_edge("technical_analyst_agent", "ticker_filter")
    workflow.add_edge("sentiment_agent", "ticker_filter")
    
    # Ensure ticker filter waits for all first tier agents including asset allocation agents
    if include_macro:
        workflow.add_edge("macro_agent", "ticker_filter")
    if include_forward_looking:
        workflow.add_edge("forward_looking_agent", "ticker_filter")
    
    # Second tier: Expensive analysts (run only on filtered tickers)
    workflow.add_node("fundamentals_agent", fundamentals_agent_filtered)
    workflow.add_node("valuation_agent", valuation_agent_filtered)
    workflow.add_node("warren_buffett_agent", warren_buffett_agent_filtered)
    workflow.add_node("bill_ackman_agent", bill_ackman_agent_filtered)
    
    # Connect filtered agents in parallel
    workflow.add_edge("ticker_filter", "fundamentals_agent")
    workflow.add_edge("ticker_filter", "valuation_agent")
    workflow.add_edge("ticker_filter", "warren_buffett_agent")
    workflow.add_edge("ticker_filter", "bill_ackman_agent")
    
    # Second filter node - filters based on 4+ agreeing signals from all 6 agents
    workflow.add_node("signal_consensus_filter", signal_consensus_filter_node)
    workflow.add_edge("fundamentals_agent", "signal_consensus_filter")
    workflow.add_edge("valuation_agent", "signal_consensus_filter")
    workflow.add_edge("warren_buffett_agent", "signal_consensus_filter")
    workflow.add_edge("bill_ackman_agent", "signal_consensus_filter")
    
    # Management nodes - risk management needs macro and forward-looking agent outputs
    workflow.add_node("risk_management_agent", risk_management_agent_filtered)
    workflow.add_node("equity_agent", equity_agent_filtered)
    
    workflow.add_edge("signal_consensus_filter", "risk_management_agent")
    
    # Asset allocation agents now flow through ticker_filter, so no direct connection to risk management needed
    
    workflow.add_edge("risk_management_agent", "equity_agent")
    
    # Only add fixed-income agent if macro or forward-looking agents are selected
    if include_macro or include_forward_looking:
        workflow.add_node("fixed_income_agent", fixed_income_agent)
        workflow.add_edge("risk_management_agent", "fixed_income_agent")
        workflow.add_edge("equity_agent", END)
        workflow.add_edge("fixed_income_agent", END)
    else:
        workflow.add_edge("equity_agent", END)
    
    workflow.set_entry_point("start_node")
    return workflow


def ticker_filter_node(state: AgentState):
    """Filter tickers based on agreement between technical and sentiment signals."""
    analyst_signals = state["data"].get("analyst_signals", {})
    tickers = state["data"].get("tickers", [])
    open_positions = state["data"].get("open_positions", set())
    
    # Get signals from technicals and sentiment agents
    tech_signals = analyst_signals.get("technical_analyst_agent", {})
    sentiment_signals = analyst_signals.get("sentiment_agent", {})
    
    # Determine which tickers have agreeing non-neutral signals
    agreeing_tickers = []
    disagreeing_tickers = []
    
    for ticker in tickers:
        tech_signal = tech_signals.get(ticker, {}).get("signal", "neutral")
        sentiment_signal = sentiment_signals.get(ticker, {}).get("signal", "neutral")
        
        # Check if both agree and have non-neutral signals OR if ticker has open positions
        if (tech_signal == sentiment_signal and tech_signal != "neutral") or ticker in open_positions:
            agreeing_tickers.append(ticker)
        else:
            disagreeing_tickers.append(ticker)
            # Set disagreeing tickers to neutral for both agents (but not for open positions)
            if ticker not in open_positions:
                if ticker in tech_signals:
                    tech_signals[ticker]["signal"] = "neutral"
                if ticker in sentiment_signals:
                    sentiment_signals[ticker]["signal"] = "neutral"
    
    # Store filtered ticker lists in state
    state["data"]["agreeing_tickers"] = agreeing_tickers
    state["data"]["disagreeing_tickers"] = disagreeing_tickers
    
    # Log the agreeing tickers for visibility
    logger.info(f"Found {len(agreeing_tickers)} agreeing tickers: {agreeing_tickers}", module="ticker_filter")
    logger.info(f"Included {len(open_positions)} open positions that bypass filtering", module="ticker_filter")
    
    return state


def fundamentals_agent_filtered(state: AgentState):
    """Run fundamentals agent only on agreeing tickers."""
    # Temporarily replace tickers with agreeing tickers
    original_tickers = state["data"]["tickers"]
    agreeing_tickers = state["data"].get("agreeing_tickers", [])
    
    if not agreeing_tickers:
        return state  # Skip if no agreeing tickers
    
    state["data"]["tickers"] = agreeing_tickers
    result = fundamentals_agent(state)
    state["data"]["tickers"] = original_tickers  # Restore original
    return result


def valuation_agent_filtered(state: AgentState):
    """Run valuation agent only on agreeing tickers."""
    original_tickers = state["data"]["tickers"]
    agreeing_tickers = state["data"].get("agreeing_tickers", [])
    
    if not agreeing_tickers:
        return state
    
    state["data"]["tickers"] = agreeing_tickers
    result = valuation_agent(state)
    state["data"]["tickers"] = original_tickers
    return result


def warren_buffett_agent_filtered(state: AgentState):
    """Run Warren Buffett agent only on agreeing tickers."""
    original_tickers = state["data"]["tickers"]
    agreeing_tickers = state["data"].get("agreeing_tickers", [])
    
    if not agreeing_tickers:
        return state
    
    state["data"]["tickers"] = agreeing_tickers
    result = warren_buffett_agent(state)
    state["data"]["tickers"] = original_tickers
    return result


def bill_ackman_agent_filtered(state: AgentState):
    """Run Bill Ackman agent only on agreeing tickers."""
    original_tickers = state["data"]["tickers"]
    agreeing_tickers = state["data"].get("agreeing_tickers", [])
    
    if not agreeing_tickers:
        return state
    
    state["data"]["tickers"] = agreeing_tickers
    result = bill_ackman_agent(state)
    state["data"]["tickers"] = original_tickers
    return result


def signal_consensus_filter_node(state: AgentState):
    """Second filter: Only pass tickers where at least 4 out of 6 signals agree (non-neutral)."""
    analyst_signals = state["data"].get("analyst_signals", {})
    agreeing_tickers = state["data"].get("agreeing_tickers", [])
    open_positions = state["data"].get("open_positions", set())
    
    # Get signals from all 6 agents
    agent_names = [
        "technical_analyst_agent",
        "sentiment_agent", 
        "fundamentals_agent",
        "valuation_agent",
        "warren_buffett_agent",
        "bill_ackman_agent"
    ]
    
    consensus_tickers = []
    filtered_out_tickers = []
    
    for ticker in agreeing_tickers:
        # Always include open positions, regardless of consensus
        if ticker in open_positions:
            consensus_tickers.append(ticker)
            continue
            
        signals = []
        for agent_name in agent_names:
            signal = analyst_signals.get(agent_name, {}).get(ticker, {}).get("signal", "neutral")
            if signal != "neutral":  # Only count non-neutral signals
                signals.append(signal)
        
        if len(signals) >= 4:  # Need at least 4 non-neutral signals
            # Check if at least 4 signals agree on direction
            bullish_count = signals.count("bullish")
            bearish_count = signals.count("bearish")
            
            if bullish_count >= 4 or bearish_count >= 4:
                consensus_tickers.append(ticker)
            else:
                filtered_out_tickers.append(ticker)
                # Set all signals to neutral for this ticker
                for agent_name in agent_names:
                    if ticker in analyst_signals.get(agent_name, {}):
                        analyst_signals[agent_name][ticker]["signal"] = "neutral"
        else:
            filtered_out_tickers.append(ticker)
            # Set all signals to neutral for this ticker
            for agent_name in agent_names:
                if ticker in analyst_signals.get(agent_name, {}):
                    analyst_signals[agent_name][ticker]["signal"] = "neutral"
    
    # Store consensus results in state
    state["data"]["consensus_tickers"] = consensus_tickers
    state["data"]["consensus_filtered_tickers"] = filtered_out_tickers
    
    # Log the consensus results
    logger.info(f"Found {len(consensus_tickers)} consensus tickers: {consensus_tickers}", module="ticker_filter")
    logger.info(f"Included {len([t for t in consensus_tickers if t in open_positions])} open positions that bypass consensus filtering", module="ticker_filter")
    
    return state


def risk_management_agent_filtered(state: AgentState):
    """Run risk management only on consensus tickers."""
    original_tickers = state["data"]["tickers"]
    consensus_tickers = state["data"].get("consensus_tickers", [])
    
    if not consensus_tickers:
        return state  # Skip if no consensus tickers
    
    state["data"]["tickers"] = consensus_tickers
    result = risk_management_agent(state)
    state["data"]["tickers"] = original_tickers  # Restore original
    return result


def equity_agent_filtered(state: AgentState):
    """Run equity agent on consensus tickers, add hold decisions for all other tickers."""
    import json
    from langchain_core.messages import HumanMessage
    
    original_tickers = state["data"]["tickers"]
    consensus_tickers = state["data"].get("consensus_tickers", [])
    disagreeing_tickers = state["data"].get("disagreeing_tickers", [])
    consensus_filtered_tickers = state["data"].get("consensus_filtered_tickers", [])
    
    # Run equity agent on consensus tickers only
    if consensus_tickers:
        # Clear previous ticker display for equity agent
        progress.update_status("equity_agent", None, "Starting analysis")
        state["data"]["tickers"] = consensus_tickers
        result = equity_agent(state)
        state["data"]["tickers"] = original_tickers
        
        # Extract decisions from equity agent's message
        equity_decisions = {}
        for message in result["messages"]:
            if hasattr(message, 'name') and message.name == "equity_agent":
                try:
                    equity_decisions = json.loads(message.content)
                except:
                    pass
                break
    else:
        result = state
        equity_decisions = {}
    
    # Add default hold decisions for all filtered tickers
    filtered_tickers = disagreeing_tickers + consensus_filtered_tickers
    for ticker in filtered_tickers:
        if ticker in disagreeing_tickers:
            reason = "Filtered out due to disagreement between technical and sentiment signals"
        else:
            reason = "Filtered out due to insufficient signal consensus (need 4+ agreeing non-neutral signals)"
        
        equity_decisions[ticker] = {
            "action": "hold",
            "quantity": 0,
            "confidence": 0.0,
            "reasoning": reason
        }
    
    # Create final message with all decisions
    message = HumanMessage(
        content=json.dumps(equity_decisions),
        name="equity_agent",
    )
    
    # Replace the equity agent's message with our merged message
    messages = [msg for msg in result["messages"] if not (hasattr(msg, 'name') and msg.name == "equity_agent")]
    
    return {
        "messages": messages + [message],
        "data": result["data"],
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash position. Defaults to 100000.0)"
    )
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=0.0,
        help="Initial margin requirement. Defaults to 0.0"
    )
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of stock ticker symbols"
    )
    ticker_group.add_argument(
        "--screen",
        action="store_true",
        help="Screen all S&P 500 tickers instead of specifying individual tickers"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 3 months before end date",
    )
    parser.add_argument(
        "--end-date",
        type=str, 
        help="End date (YYYY-MM-DD). Defaults to today"
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true", 
        help="Show reasoning from each agent"
    )
    parser.add_argument(
        "--show-agent-graph", 
        action="store_true", 
        help="Show the agent graph"
    )
    # Debugging options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging",
    )
    parser.add_argument(
        "--verbose-data",
        action="store_true",
        help="Show detailed data output (works with debug mode)",
    )

    args = parser.parse_args()

    # Set up the logger with command line arguments
    setup_logger(
        debug_mode=args.debug
    )

    # Parse tickers based on the selected mode
    if args.screen:
        tickers = get_sp500_tickers()
        logger.debug(f"Using {len(tickers)} S&P 500 tickers for screening (source: WRDS historical data)", module="main")
    else:
        # Parse tickers from comma-separated string
        tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    # Select analysts
    selected_analysts = None
    choices = questionary.checkbox(
        "Select your AI analysts.",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\n\nInstructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done to run the hedge fund.\n",
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
        print(f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}\n")

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
        # Get model info using the helper function
        model_info = get_model_info(model_choice)
        if model_info:
            model_provider = model_info.provider.value
            print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
        else:
            model_provider = "Unknown"
            print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")

    # Create the workflow with selected analysts
    workflow = create_workflow(selected_analysts)
    app = workflow.compile()

    if args.show_agent_graph:
        file_path = ""
        if selected_analysts is not None:
            for selected_analyst in selected_analysts:
                file_path += selected_analyst + "_"
            file_path += "graph.png"
        save_graph_as_png(app, file_path)

    # Validate dates if provided
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM-DD format")

    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("End date must be in YYYY-MM-DD format")

    # Set the start and end dates
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        # Calculate start date to ensure sufficient data for technical analysis
        # Technical analyst needs 127 trading days for 6-month momentum calculations
        # Using 200 calendar days to account for weekends and holidays (matches backtester.py)
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(days=200)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    # Initialize portfolio with cash amount and stock positions
    portfolio = {
        "cash": args.initial_cash,  # Initial cash amount
        "margin_requirement": args.margin_requirement,  # Initial margin requirement
        "positions": {
            ticker: {
                "long": 0,  # Number of shares held long
                "short": 0,  # Number of shares held short
                "long_cost_basis": 0.0,  # Average cost basis for long positions
                "short_cost_basis": 0.0,  # Average price at which shares were sold short
            } for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,  # Realized gains from long positions
                "short": 0.0,  # Realized gains from short positions
            } for ticker in tickers
        }
    }

    # Start the timer after LLM and analysts are selected
    start_time = time.time()

    # Run the hedge fund
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
        model_name=model_choice,
        model_provider=model_provider,
        verbose_data=args.verbose_data and args.debug,
    )

    # Stop timer after execution
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print_trading_output(result)

    print(f"\n⏱ Total elapsed time: {int(hours)}h {int(minutes)}m {seconds:.2f}s.")
