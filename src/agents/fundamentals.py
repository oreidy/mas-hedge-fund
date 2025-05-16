from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import json

from tools.api import get_financial_metrics
from utils.logger import logger


##### Fundamental Agent #####
def fundamentals_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals for multiple tickers."""

    # Get verbose_data from metadata or default to False
    verbose_data = state["metadata"].get("verbose_data", False)
    logger.debug("Accessing Fundamentals Agent", module="fundamentals_agent")

    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize fundamental analysis for each ticker
    fundamental_analysis = {}

    for ticker in tickers:
        progress.update_status("fundamentals_agent", ticker, "Fetching financial metrics")

        # Get the financial metrics
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="annual",
            limit=10,
            verbose_data = verbose_data
        )

        if not financial_metrics:
            progress.update_status("fundamentals_agent", ticker, "Failed: No financial metrics found")
            continue

        # Pull the most recent financial metrics
        metrics = financial_metrics[0]

        # Check for missing metrics and log warnings
        missing_metrics = []
        important_metrics = [
            "return_on_equity", "net_margin", "operating_margin", 
            "revenue_growth", "earnings_growth", "book_value_growth",
            "current_ratio", "debt_to_equity", "free_cash_flow_per_share", "earnings_per_share",
            "price_to_earnings_ratio", "price_to_book_ratio", "price_to_sales_ratio"
        ]

        for metric_name in important_metrics:
            if getattr(metrics, metric_name) is None:
                missing_metrics.append(metric_name)

        if missing_metrics:
            logger.warning(f"Missing financial metrics: {', '.join(missing_metrics)}. Signals default to neutral.", 
                        module="fundamentals_agent", ticker=ticker)

        # Initialize signals list for different fundamental aspects
        signals = []
        reasoning = {}

        progress.update_status("fundamentals_agent", ticker, "Analyzing profitability")
        # 1. Profitability Analysis
        return_on_equity = metrics.return_on_equity
        net_margin = metrics.net_margin
        operating_margin = metrics.operating_margin

        thresholds = [
            (return_on_equity, 0.15),  # Strong ROE above 15%
            (net_margin, 0.20),  # Healthy profit margins
            (operating_margin, 0.15),  # Strong operating efficiency
        ]
        profitability_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        if any(metric is None for metric, _ in thresholds):
            signals.append("neutral")  # Default to neutral if any metric is missing
            logger.debug("Some profitability metrics are missing, defaulting to neutral signal", 
                        module="fundamentals_agent", ticker=ticker)
        else:
            signals.append("bullish" if profitability_score >= 2 else "bearish" if profitability_score == 0 else "neutral")

        reasoning["profitability_signal"] = {
            "signal": signals[0],
            "details": (f"ROE: {return_on_equity:.2%}" if return_on_equity else "ROE: N/A") + ", " + (f"Net Margin: {net_margin:.2%}" if net_margin else "Net Margin: N/A") + ", " + (f"Op Margin: {operating_margin:.2%}" if operating_margin else "Op Margin: N/A"),
        }

        if verbose_data:
            logger.debug(
                f"Profitability Analysis:\n"
                f"  ROE: {return_on_equity if return_on_equity is not None else 'N/A'}\n"
                f"  Net Margin: {net_margin if net_margin is not None else 'N/A'}\n"
                f"  Operating Margin: {operating_margin if operating_margin is not None else 'N/A'}\n"
                f"  Score: {profitability_score}/3\n"
                f"  Signal: {signals[0]}",
                module="fundamentals_agent", ticker=ticker
            )

        progress.update_status("fundamentals_agent", ticker, "Analyzing growth")
        # 2. Growth Analysis
        revenue_growth = metrics.revenue_growth
        earnings_growth = metrics.earnings_growth
        book_value_growth = metrics.book_value_growth

        thresholds = [
            (revenue_growth, 0.10),  # 10% revenue growth
            (earnings_growth, 0.10),  # 10% earnings growth
            (book_value_growth, 0.10),  # 10% book value growth
        ]
        growth_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        if any(metric is None for metric, _ in thresholds):
            signals.append("neutral")  # Default to neutral if any metric is missing
            logger.debug("Some growth metrics are missing, defaulting to neutral signal", 
                        module="fundamentals_agent", ticker=ticker)
        else:
            signals.append("bullish" if growth_score >= 2 else "bearish" if growth_score == 0 else "neutral")

        reasoning["growth_signal"] = {
            "signal": signals[1],
            "details": (f"Revenue Growth: {revenue_growth:.2%}" if revenue_growth else "Revenue Growth: N/A") + ", " + (f"Earnings Growth: {earnings_growth:.2%}" if earnings_growth else "Earnings Growth: N/A"),
        }

        if verbose_data:
            logger.debug(
                f"Growth Analysis:\n"
                f"  Revenue Growth: {revenue_growth if revenue_growth is not None else 'N/A'}\n"
                f"  Earnings Growth: {earnings_growth if earnings_growth is not None else 'N/A'}\n"
                f"  Book Value Growth: {book_value_growth if book_value_growth is not None else 'N/A'}\n"
                f"  Score: {growth_score}/3\n"
                f"  Signal: {signals[1]}",
                module="fundamentals_agent", ticker=ticker
            )

        progress.update_status("fundamentals_agent", ticker, "Analyzing financial health")
        # 3. Financial Health
        current_ratio = metrics.current_ratio
        debt_to_equity = metrics.debt_to_equity
        free_cash_flow_per_share = metrics.free_cash_flow_per_share
        earnings_per_share = metrics.earnings_per_share

        health_score = 0
        if current_ratio and current_ratio > 1.5:  # Strong liquidity
            health_score += 1
        if debt_to_equity and debt_to_equity < 0.5:  # Conservative debt levels
            health_score += 1
        if free_cash_flow_per_share and earnings_per_share and free_cash_flow_per_share > earnings_per_share * 0.8:  # Strong FCF conversion
            health_score += 1

        if current_ratio is None or debt_to_equity is None or free_cash_flow_per_share is None or earnings_per_share is None:
            signals.append("neutral")  # Default to neutral if any metric is missing
            logger.debug("Some financial health metrics are missing, defaulting to neutral signal", 
                        module="fundamentals_agent", ticker=ticker)
        else:
            signals.append("bullish" if health_score >= 2 else "bearish" if health_score == 0 else "neutral")

        reasoning["financial_health_signal"] = {
            "signal": signals[2],
            "details": (f"Current Ratio: {current_ratio:.2f}" if current_ratio else "Current Ratio: N/A") + ", " + (f"D/E: {debt_to_equity:.2f}" if debt_to_equity else "D/E: N/A"),
        }

        if verbose_data:
            logger.debug(
                f"Financial Health Analysis:\n"
                f"  Current Ratio: {current_ratio if current_ratio is not None else 'N/A'}\n"
                f"  Debt to Equity: {debt_to_equity if debt_to_equity is not None else 'N/A'}\n"
                f"  Free Cash Flow per Share: {free_cash_flow_per_share if free_cash_flow_per_share is not None else 'N/A'}\n"
                f"  Earnings per Share: {earnings_per_share if earnings_per_share is not None else 'N/A'}\n"
                f"  Score: {health_score}/3\n"
                f"  Signal: {signals[2]}",
                module="fundamentals_agent", ticker=ticker
            )

        progress.update_status("fundamentals_agent", ticker, "Analyzing valuation ratios")
        # 4. Price to X ratios
        pe_ratio = metrics.price_to_earnings_ratio
        pb_ratio = metrics.price_to_book_ratio
        ps_ratio = metrics.price_to_sales_ratio

        thresholds = [
            (pe_ratio, 25),  # Reasonable P/E ratio
            (pb_ratio, 3),  # Reasonable P/B ratio
            (ps_ratio, 5),  # Reasonable P/S ratio
        ]
        price_ratio_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        if any(metric is None for metric, _ in thresholds):
            signals.append("neutral")  # Default to neutral if any metric is missing
            logger.debug("Some price ratio metrics are missing, defaulting to neutral signal", 
                        module="fundamentals_agent", ticker=ticker)
        else:
            signals.append("bullish" if price_ratio_score >= 2 else "bearish" if price_ratio_score == 0 else "neutral")

        reasoning["price_ratios_signal"] = {
            "signal": signals[3],
            "details": (f"P/E: {pe_ratio:.2f}" if pe_ratio else "P/E: N/A") + ", " + (f"P/B: {pb_ratio:.2f}" if pb_ratio else "P/B: N/A") + ", " + (f"P/S: {ps_ratio:.2f}" if ps_ratio else "P/S: N/A"),
        }

        if verbose_data:
            logger.debug(
                f"Price Ratios Analysis:\n"
                f"  P/E Ratio: {pe_ratio if pe_ratio is not None else 'N/A'}\n"
                f"  P/B Ratio: {pb_ratio if pb_ratio is not None else 'N/A'}\n"
                f"  P/S Ratio: {ps_ratio if ps_ratio is not None else 'N/A'}\n"
                f"  Score: {price_ratio_score}/3\n"
                f"  Signal: {signals[3]}",
                module="fundamentals_agent", ticker=ticker
            )

        progress.update_status("fundamentals_agent", ticker, "Calculating final signal")
        # Determine overall signal
        bullish_signals = signals.count("bullish")
        bearish_signals = signals.count("bearish")

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level
        total_signals = len(signals)
        confidence = round(max(bullish_signals, bearish_signals) / total_signals, 2) * 100

        fundamental_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        if verbose_data:
            logger.debug(
                f"Final Analysis:\n"
                f"  Bullish Signals: {bullish_signals}/{len(signals)}\n"
                f"  Bearish Signals: {bearish_signals}/{len(signals)}\n"
                f"  Neutral Signals: {len(signals) - bullish_signals - bearish_signals}/{len(signals)}\n"
                f"  Overall Signal: {overall_signal}\n"
                f"  Confidence: {confidence}%",
                module="fundamentals_agent", ticker=ticker
            )

        progress.update_status("fundamentals_agent", ticker, "Done")

    # Create the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(fundamental_analysis),
        name="fundamentals_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["fundamentals_agent"] = fundamental_analysis

    return {
        "messages": [message],
        "data": data,
    }
