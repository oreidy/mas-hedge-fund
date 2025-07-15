import pandas as pd
import numpy as np
import os

def summarize_performance(returns, market_returns=None, risk_free_rate=0.0):
    """
    Computes and summarizes key performance metrics for a given return series.

    Args:
        returns (pd.Series): A pandas Series of daily arithmetic returns.
        market_returns (pd.Series, optional): A pandas Series of daily market returns (e.g., SPY).
                                             Required for Beta and Alpha calculation.
        risk_free_rate (float): Annual risk-free rate. Defaults to 0.0.

    Returns:
        dict: A dictionary containing the calculated performance metrics.
    """
    summary = {}

    # Annualization factor (252 trading days in a year)
    annualization_factor = 252

    # Annualized Return (from arithmetic returns)
    print(len(returns))
    summary['Annualized Return'] = (1 + returns).prod() ** (annualization_factor / len(returns)) - 1

    # Annualized Standard Deviation
    summary['Annualized Standard Deviation'] = returns.std() * np.sqrt(annualization_factor)

    # Min and Max Daily Return
    summary['Min Daily Return'] = returns.min()
    summary['Max Daily Return'] = returns.max()

    # Skewness and Kurtosis
    summary['Skewness'] = returns.skew()
    summary['Kurtosis'] = returns.kurtosis()

    # Annualized Sharpe Ratio
    # (Annualized Return - Risk-Free Rate) / Annualized Standard Deviation
    if summary['Annualized Standard Deviation'] != 0:
        summary['Annualized Sharpe Ratio'] = (summary['Annualized Return'] - risk_free_rate) / summary['Annualized Standard Deviation']
    else:
        summary['Annualized Sharpe Ratio'] = np.nan

    # Beta and Alpha (if market returns are provided)
    if market_returns is not None:
        # Align the returns and market_returns by date
        aligned_data = pd.concat([returns, market_returns], axis=1).dropna()
        aligned_data.columns = ['portfolio', 'market']

        if not aligned_data.empty:
            # Beta = Cov(Portfolio, Market) / Var(Market)
            covariance = aligned_data['portfolio'].cov(aligned_data['market'])
            market_variance = aligned_data['market'].var()
            if market_variance != 0:
                summary['Beta'] = covariance / market_variance
            else:
                summary['Beta'] = np.nan

            # Alpha = Portfolio Return - (Risk-Free Rate + Beta * (Market Return - Risk-Free Rate))
            # Convert annual risk-free rate to daily
            daily_risk_free_rate = (1 + risk_free_rate)**(1/annualization_factor) - 1

            if 'Beta' in summary and not np.isnan(summary['Beta']):
                summary['Alpha'] = returns.mean() - (daily_risk_free_rate + summary['Beta'] * (market_returns.mean() - daily_risk_free_rate))
                summary['Annualized Alpha'] = summary['Alpha'] * annualization_factor # Simple annualization for daily alpha
            else:
                summary['Alpha'] = np.nan
                summary['Annualized Alpha'] = np.nan
        else:
            summary['Beta'] = np.nan
            summary['Alpha'] = np.nan
            summary['Annualized Alpha'] = np.nan

    return summary

def main():
    data_dir = "benchmarking/data"

    # Load benchmark returns
    try:
        spy_returns = pd.read_csv(os.path.join(data_dir, "spy_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]
        hdg_returns = pd.read_csv(os.path.join(data_dir, "hdg_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]
        portfolio_60_40_returns = pd.read_csv(os.path.join(data_dir, "60_40_portfolio_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]
    except FileNotFoundError as e:
        print(f"Error loading return data: {e}. Please ensure download_benchmarks.py has been run.")
        return

    # Placeholder for your hedge fund returns
    # Replace this with actual loading of your hedge fund returns when available
    # For now, we'll create a dummy series for demonstration
    hedge_fund_returns = pd.Series(np.random.normal(loc=0.0005, scale=0.01, size=len(spy_returns)), index=spy_returns.index)
    hedge_fund_returns.name = "Hedge Fund Returns"

    print("\n--- Performance Summary ---")

    # Summarize SPY performance
    print("\nSPY Performance:")
    spy_summary = summarize_performance(spy_returns)
    for metric, value in spy_summary.items():
        print(f"  {metric}: {value:.2%}")

    # Summarize HDG performance (vs. SPY)
    print("\nHDG Performance (vs. SPY):")
    hdg_summary = summarize_performance(hdg_returns, market_returns=spy_returns)
    for metric, value in hdg_summary.items():
        print(f"  {metric}: {value:.2%}")

    # Summarize 60/40 Portfolio performance (vs. SPY)
    print("\n60/40 Portfolio Performance (vs. SPY):")
    portfolio_60_40_summary = summarize_performance(portfolio_60_40_returns, market_returns=spy_returns)
    for metric, value in portfolio_60_40_summary.items():
        print(f"  {metric}: {value:.2%}")

    # Summarize Hedge Fund performance (vs. SPY - Placeholder Data):
    print("\nHedge Fund Performance (vs. SPY - Placeholder Data):")
    hedge_fund_summary = summarize_performance(hedge_fund_returns, market_returns=spy_returns)
    for metric, value in hedge_fund_summary.items():
        print(f"  {metric}: {value:.2%}")

if __name__ == "__main__":
    main()
