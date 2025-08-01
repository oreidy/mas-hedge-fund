import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from scipy import stats
from stargazer.stargazer import Stargazer
import statsmodels.api as sm

def load_daily_risk_free_rate():
    """Load daily risk-free rate data from CSV file."""
    try:
        rf_data = pd.read_csv("benchmarking/data/daily_risk_free_rate.csv", index_col=0, parse_dates=True)
        # Handle the case where the first column might not have a proper name
        risk_free_col = rf_data.columns[0] if len(rf_data.columns) > 0 else 'risk_free_rate'
        return rf_data[risk_free_col]
    except Exception as e:
        print(f"Warning: Could not load daily risk-free rate: {e}")
        return None

def load_fama_french_factors(factor_type="3factor"):
    """
    Load Fama-French factor data from CSV file.
    
    Args:
        factor_type: "3factor" or "5factor"
        
    Returns:
        DataFrame with factor returns or None if not available
    """
    try:
        filename = f"benchmarking/data/ff_{factor_type}_daily.csv"
        ff_data = pd.read_csv(filename, index_col=0, parse_dates=True)
        return ff_data
    except Exception as e:
        print(f"Warning: Could not load {factor_type} Fama-French factors: {e}")
        return None

def create_stargazer_table(regressions, factor_type, run_names, output_file="fama_french_results.tex"):
    """
    Create a LaTeX table using stargazer for Fama-French regression results.
    
    Args:
        regressions: List of fitted statsmodels regression models
        factor_type: String indicating "3factor" or "5factor"
        run_names: List of run names for column headers
        output_file: Output LaTeX file name
    
    Returns:
        LaTeX table string
    """
    stargazer = Stargazer(regressions)
    
    # Set appropriate covariate order based on factor type
    if factor_type == "3factor":
        stargazer.covariate_order(['Mkt-RF', 'SMB', 'HML'])
        stargazer.rename_covariates({'Mkt-RF': 'Market-RF', 'SMB': 'Size', 'HML': 'Value'})
    else:  # 5factor
        stargazer.covariate_order(['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])
        stargazer.rename_covariates({'Mkt-RF': 'Market-RF', 'SMB': 'Size', 'HML': 'Value', 
                                    'RMW': 'Profitability', 'CMA': 'Investment'})
    
    # Set custom column names for runs
    stargazer.custom_columns(run_names, [1] * len(run_names))
    
    # Generate LaTeX table
    latex_table = stargazer.render_latex()
    
    # Save to file
    with open(f"benchmarking/data/{output_file}", 'w') as f:
        f.write(latex_table)
    
    return latex_table

def fama_french_regression(excess_returns, factor_data, factor_type="3factor"):
    """
    Perform Fama-French factor regression analysis.
    
    Args:
        excess_returns: Portfolio excess returns (already risk-free rate adjusted)
        factor_data: DataFrame with factor returns
        factor_type: "3factor" or "5factor"
        
    Returns:
        Dictionary with regression results
    """
    
    # Align data by dates
    aligned_data = pd.concat([excess_returns, factor_data], axis=1).dropna()
    
    if aligned_data.empty or len(aligned_data) < 10:
        return {"Error": f"Insufficient data for {factor_type} regression"}
    
    # Prepare dependent variable (portfolio excess returns)
    y = aligned_data.iloc[:, 0].values
    
    # Prepare independent variables (factors)
    if factor_type == "3factor":
        factor_cols = ['Mkt-RF', 'SMB', 'HML']
    else:
        factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    
    # Check if all required factors are available
    available_factors = [col for col in factor_cols if col in aligned_data.columns]
    if len(available_factors) != len(factor_cols):
        return {"Error": f"Missing factors for {factor_type} model. Available: {available_factors}"}
    
    X = aligned_data[available_factors].values
    
    # Perform regression (sklearn for our calculations)
    reg = LinearRegression().fit(X, y)
    
    # Also fit statsmodels regression for stargazer
    X_df = aligned_data[available_factors].copy()  # Use DataFrame to preserve column names
    X_sm = sm.add_constant(X_df)  # Add intercept for statsmodels
    reg_sm = sm.OLS(y, X_sm).fit()
    
    # Calculate regression statistics
    y_pred = reg.predict(X)
    residuals = y - y_pred
    
    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Adjusted R-squared
    n = len(y)
    p = X.shape[1]
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    
    # Standard errors and t-statistics
    mse = ss_res / (n - p - 1)
    X_with_intercept = np.column_stack([np.ones(n), X])
    cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    se = np.sqrt(np.diag(cov_matrix))
    
    # Alpha and betas with t-stats
    alpha = reg.intercept_
    betas = reg.coef_
    alpha_tstat = alpha / se[0] if se[0] != 0 else 0
    beta_tstats = [betas[i] / se[i+1] if se[i+1] != 0 else 0 for i in range(len(betas))]
    
    # P-values (two-tailed test)
    alpha_pvalue = 2 * (1 - stats.t.cdf(abs(alpha_tstat), n - p - 1))
    beta_pvalues = [2 * (1 - stats.t.cdf(abs(t), n - p - 1)) for t in beta_tstats]
    
    # Prepare results
    results = {
        f'{factor_type.upper()} Alpha': alpha,
        f'{factor_type.upper()} Alpha (Annualized)': alpha * 252,
        f'{factor_type.upper()} Alpha T-stat': alpha_tstat,
        f'{factor_type.upper()} Alpha P-value': alpha_pvalue,
        f'{factor_type.upper()} R-squared': r_squared,
        f'{factor_type.upper()} Adj R-squared': adj_r_squared,
        f'{factor_type.upper()} Observations': n
    }
    
    # Add factor loadings
    for i, factor in enumerate(available_factors):
        results[f'{factor_type.upper()} {factor} Beta'] = betas[i]
        results[f'{factor_type.upper()} {factor} T-stat'] = beta_tstats[i]
        results[f'{factor_type.upper()} {factor} P-value'] = beta_pvalues[i]
    
    return results, reg_sm

def summarize_performance(returns, market_returns=None, daily_risk_free_rate=None):
    """
    Computes and summarizes key performance metrics for a given return series.

    Args:
        returns (pd.Series): A pandas Series of daily arithmetic returns.
        market_returns (pd.Series, optional): A pandas Series of daily market returns (e.g., SPY).
                                             Required for Beta and Alpha calculation.
        daily_risk_free_rate (pd.Series, optional): A pandas Series of daily risk-free rates.

    Returns:
        dict: A dictionary containing the calculated performance metrics.
    """
    summary = {}

    # Annualization factor (252 trading days in a year)
    annualization_factor = 252

    # Calculate excess returns if daily risk-free rate is provided
    if daily_risk_free_rate is not None:
        aligned_data = pd.concat([returns, daily_risk_free_rate], axis=1).dropna()
        aligned_data.columns = ['returns', 'risk_free']
        excess_returns = aligned_data['returns'] - aligned_data['risk_free']
        summary['Excess Returns Available'] = True
        
        # Calculate geometric averages for Sharpe ratio
        n_years = len(aligned_data) / annualization_factor
        final_portfolio_value = (1 + aligned_data['returns']).prod()
        final_rf_value = (1 + aligned_data['risk_free']).prod()
        
        avg_total_return = final_portfolio_value ** (1 / n_years) - 1
        avg_rf_return = final_rf_value ** (1 / n_years) - 1
        avg_excess_return = avg_total_return - avg_rf_return
    else:
        excess_returns = returns
        summary['Excess Returns Available'] = False
        avg_excess_return = np.nan

    print(f"Analyzing {len(returns)} return observations")
    summary['Annualized Return'] = avg_total_return if daily_risk_free_rate is not None else (1 + returns).prod() ** (annualization_factor / len(returns)) - 1

    # Annualized Standard Deviation (of excess returns)
    summary['Annualized Standard Deviation'] = excess_returns.std() * np.sqrt(annualization_factor)

    # Min and Max Daily Return (of excess returns)
    summary['Min Daily Return'] = excess_returns.min()
    summary['Max Daily Return'] = excess_returns.max()

    # Skewness and Kurtosis (of excess returns)
    summary['Skewness'] = excess_returns.skew()
    summary['Kurtosis'] = excess_returns.kurtosis()

    # Annualized Sharpe Ratio
    if summary['Annualized Standard Deviation'] != 0 and daily_risk_free_rate is not None:
        summary['Annualized Sharpe Ratio'] = avg_excess_return / summary['Annualized Standard Deviation']
    else:
        summary['Annualized Sharpe Ratio'] = np.nan

    # Beta and Alpha (if market returns are provided)
    if market_returns is not None:
        # For Beta and Alpha, we need market excess returns too
        if daily_risk_free_rate is not None:
            # Calculate market excess returns
            market_aligned = pd.concat([market_returns, daily_risk_free_rate], axis=1).dropna()
            if len(market_aligned.columns) == 2:
                market_aligned.columns = ['market_returns', 'risk_free']
                market_excess_returns = market_aligned['market_returns'] - market_aligned['risk_free']
            else:
                market_excess_returns = market_returns
        else:
            market_excess_returns = market_returns

        # Align the excess returns and market excess returns by date
        aligned_data = pd.concat([excess_returns, market_excess_returns], axis=1).dropna()
        aligned_data.columns = ['portfolio_excess', 'market_excess']

        if not aligned_data.empty:
            # Beta = Cov(Portfolio Excess Returns, Market Excess Returns) / Var(Market Excess Returns)
            covariance = aligned_data['portfolio_excess'].cov(aligned_data['market_excess'])
            market_variance = aligned_data['market_excess'].var()
            if market_variance != 0:
                summary['Beta'] = covariance / market_variance
            else:
                summary['Beta'] = np.nan

            # Alpha = Mean Portfolio Excess Return - Beta * Mean Market Excess Return
            if 'Beta' in summary and not np.isnan(summary['Beta']):
                summary['Alpha'] = aligned_data['portfolio_excess'].mean() - summary['Beta'] * aligned_data['market_excess'].mean()
                summary['Annualized Alpha'] = summary['Alpha'] * annualization_factor
            else:
                summary['Alpha'] = np.nan
                summary['Annualized Alpha'] = np.nan
        else:
            summary['Beta'] = np.nan
            summary['Alpha'] = np.nan
            summary['Annualized Alpha'] = np.nan

    # Fama-French Factor Analysis
    regression_models = {}
    if daily_risk_free_rate is not None:
        # Load factor data
        ff_3factor = load_fama_french_factors("3factor")
        ff_5factor = load_fama_french_factors("5factor")
        
        if ff_3factor is not None:
            # Perform 3-factor regression
            ff_3factor_results, ff_3factor_model = fama_french_regression(excess_returns, ff_3factor, "3factor")
            summary.update(ff_3factor_results)
            regression_models['3factor'] = ff_3factor_model
        
        if ff_5factor is not None:
            # Perform 5-factor regression
            ff_5factor_results, ff_5factor_model = fama_french_regression(excess_returns, ff_5factor, "5factor")
            summary.update(ff_5factor_results)
            regression_models['5factor'] = ff_5factor_model
    
    # Add regression models to summary for later use
    summary['regression_models'] = regression_models

    return summary

def main():
    data_dir = "benchmarking/data"

    # Load daily risk-free rate
    daily_rf_rate = load_daily_risk_free_rate()
    
    # Load benchmark returns
    try:
        spy_returns = pd.read_csv(os.path.join(data_dir, "spy_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]
        hdg_returns = pd.read_csv(os.path.join(data_dir, "hdg_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]
        agg_returns = pd.read_csv(os.path.join(data_dir, "agg_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]
        portfolio_60_40_returns = pd.read_csv(os.path.join(data_dir, "60_40_portfolio_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]
    except FileNotFoundError as e:
        print(f"Error loading return data: {e}. Please ensure download_benchmarks.py has been run.")
        return

    # Load all hedge fund returns files
    hedge_fund_files = [f for f in os.listdir(data_dir) if f.startswith("mas_hedge_fund_returns_") and f.endswith(".csv")]
    
    if not hedge_fund_files:
        print("No hedge fund returns files found. Please ensure you have files starting with 'mas_hedge_fund_returns_' in benchmarking/data/")
        return
    
    hedge_fund_runs = {}
    for file in sorted(hedge_fund_files):
        try:
            run_name = file.replace("mas_hedge_fund_returns_", "").replace(".csv", "")
            returns = pd.read_csv(os.path.join(data_dir, file), index_col=0, parse_dates=True).iloc[:, 0]
            returns.name = f"HF Run {run_name}"
            hedge_fund_runs[run_name] = returns
            print(f"Loaded hedge fund run: {run_name}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not hedge_fund_runs:
        print("No valid hedge fund returns files could be loaded.")
        return

    print("\n--- Performance Summary ---")
    if daily_rf_rate is not None:
        print("Using daily risk-free rates for excess return calculations")
    else:
        print("Warning: Daily risk-free rates not available, using raw returns")

    # Summarize SPY performance
    print("\nSPY Performance:")
    spy_summary = summarize_performance(spy_returns, daily_risk_free_rate=daily_rf_rate)
    for metric, value in spy_summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"  {metric}: {value:.4f}" if 'Return' in metric or 'Ratio' in metric or 'Alpha' in metric else f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # Summarize HDG performance (vs. SPY)
    print("\nHDG Performance (vs. SPY):")
    hdg_summary = summarize_performance(hdg_returns, market_returns=spy_returns, daily_risk_free_rate=daily_rf_rate)
    for metric, value in hdg_summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"  {metric}: {value:.4f}" if 'Return' in metric or 'Ratio' in metric or 'Alpha' in metric else f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # Summarize 60/40 Portfolio performance (vs. SPY)
    print("\n60/40 Portfolio Performance (vs. SPY):")
    portfolio_60_40_summary = summarize_performance(portfolio_60_40_returns, market_returns=spy_returns, daily_risk_free_rate=daily_rf_rate)
    for metric, value in portfolio_60_40_summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"  {metric}: {value:.4f}" if 'Return' in metric or 'Ratio' in metric or 'Alpha' in metric else f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # Analyze each hedge fund run individually
    hedge_fund_summaries = {}
    print(f"\n--- Individual Hedge Fund Run Analysis ({len(hedge_fund_runs)} runs) ---")
    
    for run_name, returns in hedge_fund_runs.items():
        print(f"\nHedge Fund Run {run_name} Performance (vs. SPY):")
        summary = summarize_performance(returns, market_returns=spy_returns, daily_risk_free_rate=daily_rf_rate)
        hedge_fund_summaries[run_name] = summary
        
        for metric, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    # Simple summary across runs
    print(f"\nAnalyzed {len(hedge_fund_runs)} hedge fund backtesting runs.")
    
    # Generate LaTeX tables for Fama-French regressions using stargazer
    print("\nGenerating LaTeX tables for Fama-French regressions...")
    
    # Collect regression models for LaTeX table
    models_3factor = []
    models_5factor = []
    run_names_list = []
    
    for run_name, summary in hedge_fund_summaries.items():
        run_names_list.append(f"Run {run_name}")
        
        if 'regression_models' in summary:
            if '3factor' in summary['regression_models']:
                models_3factor.append(summary['regression_models']['3factor'])
            if '5factor' in summary['regression_models']:
                models_5factor.append(summary['regression_models']['5factor'])
    
    # Create stargazer tables
    if models_3factor:
        print(f"Creating 3-factor LaTeX table for {len(models_3factor)} runs...")
        latex_3factor = create_stargazer_table(models_3factor, "3factor", 
                                             run_names_list[:len(models_3factor)], "fama_french_3factor.tex")
        print("3-factor LaTeX table saved to benchmarking/data/fama_french_3factor.tex")
    
    if models_5factor:
        print(f"Creating 5-factor LaTeX table for {len(models_5factor)} runs...")
        latex_5factor = create_stargazer_table(models_5factor, "5factor", 
                                             run_names_list[:len(models_5factor)], "fama_french_5factor.tex")
        print("5-factor LaTeX table saved to benchmarking/data/fama_french_5factor.tex")

if __name__ == "__main__":
    main()
