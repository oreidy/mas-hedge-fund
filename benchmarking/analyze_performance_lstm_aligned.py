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

def create_stargazer_table_lstm_aligned(regressions, factor_type, run_names, output_file="fama_french_results_lstm_aligned.tex"):
    """
    Create a LaTeX table using stargazer for Fama-French regression results aligned to LSTM period.
    
    Args:
        regressions: List of fitted statsmodels regression models
        factor_type: String indicating "3factor" or "5factor"
        run_names: List of run names for column headers
        output_file: Output LaTeX file name
    
    Returns:
        LaTeX table string
    """
    stargazer = Stargazer(regressions)
    
    # Set appropriate covariate order based on factor type (including constant/alpha)
    if factor_type == "3factor":
        stargazer.covariate_order(['const', 'Mkt-RF', 'SMB', 'HML'])
        stargazer.rename_covariates({'const': 'Alpha', 'Mkt-RF': 'Market-RF', 'SMB': 'Size', 'HML': 'Value'})
    else:  # 5factor
        stargazer.covariate_order(['const', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])
        stargazer.rename_covariates({'const': 'Alpha', 'Mkt-RF': 'Market-RF', 'SMB': 'Size', 'HML': 'Value', 
                                    'RMW': 'Profitability', 'CMA': 'Investment'})
    
    # Fix column names to use 60/40 instead of 60/40 Portfolio
    fixed_run_names = []
    for name in run_names:
        if name == '60/40 Portfolio':
            fixed_run_names.append('60/40')
        else:
            fixed_run_names.append(name)
    
    # Set custom column names for runs
    stargazer.custom_columns(fixed_run_names, [1] * len(fixed_run_names))
    
    # Generate LaTeX table
    latex_table = stargazer.render_latex()
    
    # Post-process the LaTeX table to make modifications
    lines = latex_table.split('\n')
    processed_lines = []
    
    for line in lines:
        # Change "Dependent variable: y" to "Dependent variable: Excess Returns"
        if 'Dependent variable: y' in line:
            line = line.replace('Dependent variable: y', 'Dependent variable: Excess Returns')
        
        # Process Residual Std. Error line to put degrees of freedom on next line
        if 'Residual Std. Error' in line:
            # Split the line to extract the values and df parts
            parts = line.split(' & ')
            new_parts = []
            df_parts = []
            
            for i, part in enumerate(parts):
                if i == 0:  # First part is "Residual Std. Error"
                    new_parts.append(part)
                    df_parts.append('')
                else:
                    # Extract the value and df part
                    if '(df=' in part:
                        value_part = part.split(' (df=')[0]
                        df_part = '(' + part.split(' (df=')[1]
                        new_parts.append(value_part)
                        df_parts.append(df_part)
                    else:
                        new_parts.append(part)
                        df_parts.append('')
            
            # Create the main line with just values
            processed_lines.append(' & '.join(new_parts) + ' \\\\')
            # Create the df line
            df_line = ' & '.join(df_parts)
            processed_lines.append(df_line)
            continue
        
        # Process F Statistic line similarly
        if 'F Statistic' in line:
            parts = line.split(' & ')
            new_parts = []
            df_parts = []
            
            for i, part in enumerate(parts):
                if i == 0:  # First part is "F Statistic"
                    new_parts.append(part)
                    df_parts.append('')
                else:
                    # Extract the value and df part
                    if '(df=' in part:
                        value_part = part.split(' (df=')[0]
                        df_part = '(' + part.split(' (df=')[1]
                        new_parts.append(value_part)
                        df_parts.append(df_part)
                    else:
                        new_parts.append(part)
                        df_parts.append('')
            
            # Create the main line with just values
            processed_lines.append(' & '.join(new_parts) + ' \\\\')
            # Create the df line
            df_line = ' & '.join(df_parts)
            processed_lines.append(df_line)
            continue
        
        processed_lines.append(line)
    
    # Add caption and label at the end (before \end{table})
    final_lines = []
    for i, line in enumerate(processed_lines):
        if line.strip() == '\\end{table}':
            # Add caption before \end{table}
            if factor_type == "3factor":
                caption = ("\\caption[Fama-French 3-Factor Model Results (LSTM Period)]{This table reports the results of Fama-French "
                          "3-factor model regressions for excess returns of investment strategies during the LSTM backtest period "
                          "(March 2023 - June 2025). The dependent variable "
                          "is daily excess returns (total returns minus the daily risk-free rate) for each strategy. "
                          "Independent variables are the three Fama-French factors: Market-RF (market excess return), "
                          "Size (SMB, small minus big), and Value (HML, high minus low book-to-market). Alpha represents "
                          "the intercept coefficient (risk-adjusted excess return). "
                          "SPY represents the S\\&P 500 ETF, HDG is a hedge fund strategy ETF, 60/40 is a traditional "
                          "portfolio of 60\\% SPY and 40\\% AGG (bond ETF), MAS1-MAS5 are implementations "
                          "of the multi-agent system hedge fund, and LSTM is the LSTM-based trading strategy. Standard errors are reported in parentheses below "
                          "coefficient estimates. Degrees of freedom are shown below test statistics. "
                          "Statistical significance is indicated by: * p < 0.1, ** p < 0.05, *** p < 0.01.}")
            else:  # 5factor
                caption = ("\\caption[Fama-French 5-Factor Model Results (LSTM Period)]{This table reports the results of Fama-French "
                          "5-factor model regressions for excess returns of investment strategies during the LSTM backtest period "
                          "(March 2023 - June 2025). The dependent variable "
                          "is daily excess returns (total returns minus the daily risk-free rate) for each strategy. "
                          "Independent variables are the five Fama-French factors: Market-RF (market excess return), "
                          "Size (SMB, small minus big), Value (HML, high minus low book-to-market), "
                          "Profitability (RMW, robust minus weak), and Investment (CMA, conservative minus aggressive). "
                          "Alpha represents the intercept coefficient (risk-adjusted excess return). "
                          "SPY represents the S\\&P 500 ETF, HDG is a hedge fund strategy ETF, 60/40 is a traditional "
                          "portfolio of 60\\% SPY and 40\\% AGG (bond ETF), MAS1-MAS5 are implementations "
                          "of the multi-agent system hedge fund, and LSTM is the LSTM-based trading strategy. Standard errors are reported in parentheses below "
                          "coefficient estimates. Degrees of freedom are shown below test statistics. "
                          "Statistical significance is indicated by: * p < 0.1, ** p < 0.05, *** p < 0.01.}")
            
            final_lines.append(caption)
            final_lines.append(f"\\label{{tab:fama_french_{factor_type}_lstm_aligned}}")
        
        final_lines.append(line)
    
    latex_table = '\n'.join(final_lines)
    
    # Save to file
    with open(f"benchmarking/data/{output_file}", 'w') as f:
        f.write(latex_table)
    
    return latex_table

def create_performance_latex_table(summaries, output_file="performance_summary_lstm_aligned.tex"):
    """
    Create a LaTeX table for standard performance metrics.
    
    Args:
        summaries: Dictionary of performance summaries {asset_name: summary_dict}
        output_file: Output LaTeX file name
    
    Returns:
        LaTeX table string
    """
    # Define the metrics we want to include in the table
    metrics_to_include = [
        ('Annualized Return', 'Ann. Return', '%.4f'),
        ('Annualized Standard Deviation', 'Ann. Std Dev', '%.4f'),
        ('Annualized Sharpe Ratio', 'Sharpe Ratio', '%.4f'),
        ('Beta', 'Beta', '%.4f'),
        ('Annualized Alpha', 'Ann. Alpha', '%.4f'),
        ('Min Daily Return', 'Min Return', '%.4f'),
        ('Max Daily Return', 'Max Return', '%.4f'),
        ('Skewness', 'Skewness', '%.4f'),
        ('Kurtosis', 'Kurtosis', '%.4f')
    ]
    
    # Define the desired column order: benchmarks, LSTM, then MAS strategies
    desired_order = ['SPY', 'HDG', '60/40 Portfolio', 'LSTM']
    
    # Add MAS strategies in order
    mas_strategies = []
    for key in summaries.keys():
        if key.startswith('LLM_20250726'):
            mas_strategies.append(('MAS1', key))
        elif key.startswith('LLM_20250731'):
            mas_strategies.append(('MAS2', key))
        elif key.startswith('LLM_20250805'):
            mas_strategies.append(('MAS3', key))
        elif key.startswith('LLM_20250806'):
            mas_strategies.append(('MAS4', key))
        elif key.startswith('LLM_20250808'):
            mas_strategies.append(('MAS5', key))
    
    # Sort MAS strategies by their number
    mas_strategies.sort(key=lambda x: x[0])
    
    # Create ordered list of asset keys and their display names
    ordered_assets = []
    ordered_display_names = []
    
    # Add benchmarks and LSTM in fixed order
    for asset in desired_order:
        if asset in summaries:
            ordered_assets.append(asset)
            if asset == '60/40 Portfolio':
                ordered_display_names.append('60/40')
            else:
                ordered_display_names.append(asset)
    
    # Add MAS strategies
    for display_name, asset_key in mas_strategies:
        ordered_assets.append(asset_key)
        ordered_display_names.append(display_name)
    
    # Start building LaTeX table
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\begin{tabular}{l" + "c" * len(ordered_assets) + "}",
        "\\toprule"
    ]
    
    # Create header row with ordered display names
    header = "Metric & " + " & ".join(ordered_display_names) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    
    # Add each metric row
    for metric_key, metric_display, format_str in metrics_to_include:
        row_values = []
        for asset_name in ordered_assets:
            summary = summaries[asset_name]
            if metric_key in summary and summary[metric_key] is not None:
                if isinstance(summary[metric_key], (int, float)) and not np.isnan(summary[metric_key]):
                    row_values.append(format_str % summary[metric_key])
                else:
                    row_values.append("N/A")
            else:
                row_values.append("N/A")
        
        row = metric_display + " & " + " & ".join(row_values) + " \\\\"
        latex_lines.append(row)
    
    # Close table with caption and label at the bottom
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption[Performance Summary Statistics (LSTM Period)]{This table reports annualized performance metrics for three benchmark strategies: SPY (S\\&P 500 ETF), HDG (hedge fund strategy ETF), and a traditional 60/40 portfolio consisting of 60 percent SPY and 40 percent AGG (a bond ETF). These are compared with MAS1-MAS5 (five implementations of a multi-agent system hedge fund) and the LSTM trading strategy. All data is aligned to the LSTM backtest period (March 2023 - June 2025) for fair comparison. All returns are total returns, calculated by adding the daily risk-free rate to the daily excess returns. Reported metrics include Annual Return, Annualized Standard Deviation, Sharpe Ratio, Beta relative to SPY where applicable, Annual Alpha, Minimum and Maximum Daily Returns, Skewness, and Kurtosis.}",
        "\\label{tab:performance_summary_lstm_aligned}",
        "\\end{table}"
    ])
    
    latex_table = "\n".join(latex_lines)
    
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

    # Load LSTM data first to get the date range
    lstm_file = "lstm_walkforward_returns_20250811_095230.csv"
    try:
        lstm_returns = pd.read_csv(os.path.join(data_dir, lstm_file), index_col=0, parse_dates=True).iloc[:, 0]
        print(f"LSTM backtest period: {lstm_returns.index.min()} to {lstm_returns.index.max()}")
        print(f"LSTM observations: {len(lstm_returns)}")
        lstm_date_range = lstm_returns.index
    except FileNotFoundError as e:
        print(f"Error loading LSTM data: {e}")
        return

    # Load daily risk-free rate and align to LSTM dates
    daily_rf_rate = load_daily_risk_free_rate()
    if daily_rf_rate is not None:
        daily_rf_rate = daily_rf_rate.reindex(lstm_date_range).dropna()
        print(f"Risk-free rate observations aligned to LSTM period: {len(daily_rf_rate)}")
    
    # Load benchmark returns and align to LSTM dates
    try:
        spy_returns = pd.read_csv(os.path.join(data_dir, "spy_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]
        hdg_returns = pd.read_csv(os.path.join(data_dir, "hdg_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]
        agg_returns = pd.read_csv(os.path.join(data_dir, "agg_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]
        portfolio_60_40_returns = pd.read_csv(os.path.join(data_dir, "60_40_portfolio_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]
        
        # Align all benchmark data to LSTM date range
        spy_returns = spy_returns.reindex(lstm_date_range).dropna()
        hdg_returns = hdg_returns.reindex(lstm_date_range).dropna()
        portfolio_60_40_returns = portfolio_60_40_returns.reindex(lstm_date_range).dropna()
        
        print(f"SPY observations aligned to LSTM period: {len(spy_returns)}")
        print(f"HDG observations aligned to LSTM period: {len(hdg_returns)}")
        print(f"60/40 observations aligned to LSTM period: {len(portfolio_60_40_returns)}")
        
    except FileNotFoundError as e:
        print(f"Error loading benchmark data: {e}. Please ensure download_benchmarks.py has been run.")
        return

    # Load all MAS strategy returns files and align to LSTM dates
    hedge_fund_files = [f for f in os.listdir(data_dir) if f.startswith("mas_hedge_fund_returns_") and f.endswith(".csv")]
    
    if not hedge_fund_files:
        print("No MAS strategy returns files found.")
        return
    
    hedge_fund_runs = {}
    
    # Load MAS hedge fund runs and align to LSTM dates
    for file in sorted(hedge_fund_files):
        try:
            run_name = file.replace("mas_hedge_fund_returns_", "").replace(".csv", "")
            returns = pd.read_csv(os.path.join(data_dir, file), index_col=0, parse_dates=True).iloc[:, 0]
            # Align to LSTM date range
            returns = returns.reindex(lstm_date_range).dropna()
            returns.name = f"LLM Run {run_name}"
            hedge_fund_runs[f"LLM_{run_name}"] = returns
            print(f"Loaded MAS hedge fund run: {run_name}, observations aligned to LSTM period: {len(returns)}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not hedge_fund_runs:
        print("No valid MAS hedge fund returns files could be loaded.")
        return

    print("\n--- Performance Summary (LSTM-Aligned Period) ---")
    if daily_rf_rate is not None:
        print("Using daily risk-free rates for excess return calculations")
    else:
        print("Warning: Daily risk-free rates not available, using raw returns")

    # Analyze benchmark performance (including Fama-French)
    benchmark_summaries = {}
    
    # Summarize SPY performance
    print("\nSPY Performance (LSTM period):")
    spy_summary = summarize_performance(spy_returns, daily_risk_free_rate=daily_rf_rate)
    benchmark_summaries['SPY'] = spy_summary
    for metric, value in spy_summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"  {metric}: {value:.4f}" if 'Return' in metric or 'Ratio' in metric or 'Alpha' in metric else f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # Summarize HDG performance (vs. SPY)
    print("\nHDG Performance (vs. SPY, LSTM period):")
    hdg_summary = summarize_performance(hdg_returns, market_returns=spy_returns, daily_risk_free_rate=daily_rf_rate)
    benchmark_summaries['HDG'] = hdg_summary
    for metric, value in hdg_summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"  {metric}: {value:.4f}" if 'Return' in metric or 'Ratio' in metric or 'Alpha' in metric else f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # Summarize 60/40 Portfolio performance (vs. SPY)
    print("\n60/40 Portfolio Performance (vs. SPY, LSTM period):")
    portfolio_60_40_summary = summarize_performance(portfolio_60_40_returns, market_returns=spy_returns, daily_risk_free_rate=daily_rf_rate)
    benchmark_summaries['60/40 Portfolio'] = portfolio_60_40_summary
    for metric, value in portfolio_60_40_summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"  {metric}: {value:.4f}" if 'Return' in metric or 'Ratio' in metric or 'Alpha' in metric else f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # Analyze each MAS strategy run individually (aligned to LSTM period)
    hedge_fund_summaries = {}
    print(f"\n--- Individual MAS Strategy Run Analysis (LSTM-aligned period, {len(hedge_fund_runs)} runs) ---")
    
    for run_name, returns in hedge_fund_runs.items():
        print(f"\nMAS Strategy Run {run_name} Performance (vs. SPY, LSTM period):")
        summary = summarize_performance(returns, market_returns=spy_returns, daily_risk_free_rate=daily_rf_rate)
        hedge_fund_summaries[run_name] = summary
        
        for metric, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    # Analyze LSTM performance
    print(f"\n--- LSTM Strategy Performance ---")
    print(f"\nLSTM Strategy Performance (vs. SPY):")
    lstm_summary = summarize_performance(lstm_returns, market_returns=spy_returns, daily_risk_free_rate=daily_rf_rate)
    lstm_summaries = {"LSTM": lstm_summary}
    
    for metric, value in lstm_summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Simple summary across runs
    mas_runs = len([k for k in hedge_fund_runs.keys() if k.startswith("LLM_")])
    print(f"\nAnalyzed {mas_runs} MAS hedge fund runs and 1 LSTM trading run, all aligned to LSTM backtest period.")
    
    # Generate LaTeX tables
    print("\nGenerating LSTM-aligned LaTeX table...")
    
    # Create performance summary table for all assets (benchmarks + MAS strategies + LSTM)
    all_summaries = {**benchmark_summaries, **hedge_fund_summaries, **lstm_summaries}
    print(f"Creating performance summary LaTeX table for {len(all_summaries)} assets (LSTM-aligned period)...")
    performance_latex = create_performance_latex_table(all_summaries, "performance_summary_lstm_aligned.tex")
    print("Performance summary LaTeX table saved to benchmarking/data/performance_summary_lstm_aligned.tex")

if __name__ == "__main__":
    main()