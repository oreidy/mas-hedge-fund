import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os


def load_daily_risk_free_rate():
    """Load daily risk-free rate data from CSV file."""
    try:
        rf_data = pd.read_csv("benchmarking/data/daily_risk_free_rate.csv", index_col=0, parse_dates=True)
        risk_free_col = rf_data.columns[0] if len(rf_data.columns) > 0 else 'risk_free_rate'
        return rf_data[risk_free_col]
    except Exception as e:
        print(f"Warning: Could not load daily risk-free rate: {e}")
        return None


def load_benchmark_data():
    """Load benchmark return data."""
    data_dir = "benchmarking/data"
    
    try:
        spy_returns = pd.read_csv(os.path.join(data_dir, "spy_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]
        hdg_returns = pd.read_csv(os.path.join(data_dir, "hdg_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]
        portfolio_60_40_returns = pd.read_csv(os.path.join(data_dir, "60_40_portfolio_returns.csv"), index_col=0, parse_dates=True).iloc[:, 0]
        
        return {
            'SPY': spy_returns,
            'HDG': hdg_returns,
            '60/40': portfolio_60_40_returns
        }
    except FileNotFoundError as e:
        print(f"Error loading benchmark data: {e}")
        return None


def load_mas_hedge_fund_data():
    """Load MAS hedge fund return data."""
    data_dir = "benchmarking/data"
    
    hedge_fund_files = [f for f in os.listdir(data_dir) if f.startswith("mas_hedge_fund_returns_") and f.endswith(".csv")]
    
    if not hedge_fund_files:
        print("No MAS hedge fund returns files found.")
        return None
    
    hedge_fund_runs = {}
    
    for file in sorted(hedge_fund_files):
        try:
            run_name = file.replace("mas_hedge_fund_returns_", "").replace(".csv", "")
            returns = pd.read_csv(os.path.join(data_dir, file), index_col=0, parse_dates=True).iloc[:, 0]
            
            # Create friendly names for the runs
            if run_name.startswith('20250726'):
                display_name = 'MAS1'
            elif run_name.startswith('20250731'):
                display_name = 'MAS2'
            else:
                display_name = f'MAS_{run_name}'
            
            hedge_fund_runs[display_name] = returns
            print(f"Loaded MAS hedge fund run: {display_name}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return hedge_fund_runs


def calculate_excess_returns(returns, daily_rf_rate):
    """Calculate excess returns by subtracting risk-free rate."""
    if daily_rf_rate is None:
        return returns
    
    aligned_data = pd.concat([returns, daily_rf_rate], axis=1).dropna()
    aligned_data.columns = ['returns', 'risk_free']
    excess_returns = aligned_data['returns'] - aligned_data['risk_free']
    
    return excess_returns


def calculate_cumulative_returns(excess_returns, start_value=100):
    """Calculate cumulative returns starting from specified value."""
    # Add 1 to excess returns and calculate cumulative product
    cumulative = (1 + excess_returns).cumprod() * start_value
    return cumulative


def align_data_to_common_start(data_dict):
    """
    Align all data series to start from the latest common start date.
    
    Args:
        data_dict: Dictionary of pandas Series with date indices
        
    Returns:
        Dictionary of aligned pandas Series
    """
    # Find the latest start date among all series
    start_dates = [series.index.min() for series in data_dict.values()]
    common_start_date = max(start_dates)
    
    print(f"Aligning all data to start from: {common_start_date}")
    
    # Align all series to start from the common date
    aligned_data = {}
    for name, series in data_dict.items():
        aligned_series = series[series.index >= common_start_date]
        aligned_data[name] = aligned_series
        print(f"  {name}: {len(aligned_series)} observations from {aligned_series.index.min()} to {aligned_series.index.max()}")
    
    return aligned_data


def plot_mas_runs_comparison(mas_data, daily_rf_rate, save_path=None):
    """
    Create a plot comparing all MAS hedge fund runs.
    
    Args:
        mas_data: Dictionary of MAS hedge fund returns
        daily_rf_rate: Daily risk-free rate series
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # LaTeX-style colors: darker blue and red for MAS runs
    mas_colors = ["#0050a0", "#a30000", '#006400', '#4b0082']  # Dark blue, dark red, dark green, dark purple
    
    for i, (run_name, returns) in enumerate(mas_data.items()):
        # Calculate excess returns
        excess_returns = calculate_excess_returns(returns, daily_rf_rate)
        
        # Calculate cumulative returns starting at 100
        cumulative = calculate_cumulative_returns(excess_returns, start_value=100)
        
        plt.plot(cumulative.index, cumulative.values, 
                label=run_name, linewidth=2, color=mas_colors[i % len(mas_colors)])
    
    # LaTeX-style formatting
    plt.xlabel('Date', fontsize=12, color='black')
    plt.ylabel('Cumulative Excess Returns (Base = 100)', fontsize=12, color='black')
    plt.legend(loc='upper left', fontsize=11, frameon=True, fancybox=False, edgecolor='black')
    plt.grid(True, alpha=0.3, color='gray')
    
    # Make axes black
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(colors='black', which='both')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    
    plt.tight_layout()
    
    # Format x-axis to show years
    plt.gca().tick_params(axis='x', rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close()  # Close figure to free memory


def plot_benchmarks_vs_mas(benchmark_data, mas_data, daily_rf_rate, save_path=None):
    """
    Create a plot comparing benchmarks with MAS hedge fund runs.
    
    Args:
        benchmark_data: Dictionary of benchmark returns
        mas_data: Dictionary of MAS hedge fund returns
        daily_rf_rate: Daily risk-free rate series
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Align all data to start from the same date (latest start date among all series)
    all_data = {**benchmark_data, **mas_data}
    aligned_data = align_data_to_common_start(all_data)
    
    # Split back into benchmarks and MAS data
    aligned_benchmarks = {k: v for k, v in aligned_data.items() if k in benchmark_data}
    aligned_mas = {k: v for k, v in aligned_data.items() if k in mas_data}
    
    # LaTeX-style color scheme with better contrast for benchmarks
    benchmark_colors = ["#2e2e2e", "#747474", "#A6A6A6"]  # Darker gray, medium gray, lighter gray
    benchmark_styles = ['-', '-', '-']  # Solid, dash-dot, dotted for better distinction
    mas_colors = ["#0050a0", "#a30000"]  # Dark blue, dark red for MAS runs
    
    # Plot benchmarks with different line styles and better contrast
    for i, (bench_name, returns) in enumerate(aligned_benchmarks.items()):
        excess_returns = calculate_excess_returns(returns, daily_rf_rate)
        cumulative = calculate_cumulative_returns(excess_returns, start_value=100)
        
        plt.plot(cumulative.index, cumulative.values, 
                label=bench_name, linewidth=1.5, 
                linestyle=benchmark_styles[i % len(benchmark_styles)],
                color=benchmark_colors[i % len(benchmark_colors)])
    
    # Plot MAS runs with solid lines
    for i, (run_name, returns) in enumerate(aligned_mas.items()):
        excess_returns = calculate_excess_returns(returns, daily_rf_rate)
        cumulative = calculate_cumulative_returns(excess_returns, start_value=100)
        
        plt.plot(cumulative.index, cumulative.values, 
                label=run_name, linewidth=2,
                color=mas_colors[i % len(mas_colors)])
    
    # LaTeX-style formatting
    plt.xlabel('Date', fontsize=12, color='black')
    plt.ylabel('Cumulative Excess Returns (Base = 100)', fontsize=12, color='black')
    plt.legend(loc='upper left', fontsize=11, frameon=True, fancybox=False, edgecolor='black')
    plt.grid(True, alpha=0.3, color='gray')
    
    # Make axes black
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(colors='black', which='both')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    
    plt.tight_layout()
    
    # Format x-axis to show years
    plt.gca().tick_params(axis='x', rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close()  # Close figure to free memory


def main():
    """Main function to create both plots."""
    print("Loading data...")
    
    # Load risk-free rate
    daily_rf_rate = load_daily_risk_free_rate()
    if daily_rf_rate is None:
        print("Warning: No risk-free rate data available. Using raw returns instead of excess returns.")
    
    # Load benchmark data
    benchmark_data = load_benchmark_data()
    if benchmark_data is None:
        print("Error: Could not load benchmark data.")
        return
    
    # Load MAS hedge fund data
    mas_data = load_mas_hedge_fund_data()
    if mas_data is None:
        print("Error: Could not load MAS hedge fund data.")
        return
    
    print(f"Loaded {len(benchmark_data)} benchmarks and {len(mas_data)} MAS runs.")
    
    # Create output directory if it doesn't exist
    output_dir = "benchmarking/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Plot 1: MAS runs comparison
    print("\nGenerating Plot 1: MAS Hedge Fund Runs Comparison...")
    plot_mas_runs_comparison(
        mas_data, 
        daily_rf_rate, 
        save_path=os.path.join(output_dir, "mas_runs_comparison.png")
    )
    
    # Generate Plot 2: Benchmarks vs MAS runs
    print("\nGenerating Plot 2: Benchmarks vs MAS Hedge Fund Runs...")
    plot_benchmarks_vs_mas(
        benchmark_data, 
        mas_data, 
        daily_rf_rate, 
        save_path=os.path.join(output_dir, "benchmarks_vs_mas.png")
    )
    
    print("\nPlot generation completed!")


if __name__ == "__main__":
    main()