#!/usr/bin/env python3
"""Plot cumulative LSTM portfolio returns to visualize the trajectory."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_lstm_returns():
    """Plot the cumulative LSTM returns to see what's happening."""
    
    # Load the CSV file
    df = pd.read_csv('benchmarking/data/lstm_walkforward_returns_20250806_102327.csv')
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate cumulative returns
    df['cumulative_return'] = (1 + df['return']).cumprod() - 1
    df['portfolio_value'] = 100000 * (1 + df['cumulative_return'])
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Cumulative Returns
    ax1.plot(df['date'], df['cumulative_return'] * 100, 'b-', linewidth=1)
    ax1.set_title('LSTM Walk-Forward Cumulative Returns', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 2: Portfolio Value
    ax2.plot(df['date'], df['portfolio_value'], 'g-', linewidth=1)
    ax2.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=100000, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
    ax2.legend()
    
    # Format y-axis for portfolio value
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 3: Daily Returns
    ax3.plot(df['date'], df['return'] * 100, 'r-', linewidth=0.5, alpha=0.7)
    ax3.set_title('Daily Returns', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Daily Return (%)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Highlight extreme returns
    extreme_positive = df[df['return'] > 0.1]
    extreme_negative = df[df['return'] < -0.1]
    
    if len(extreme_positive) > 0:
        ax3.scatter(extreme_positive['date'], extreme_positive['return'] * 100, 
                   color='green', s=20, alpha=0.8, label=f'Returns > +10% ({len(extreme_positive)})')
    
    if len(extreme_negative) > 0:
        ax3.scatter(extreme_negative['date'], extreme_negative['return'] * 100, 
                   color='red', s=20, alpha=0.8, label=f'Returns < -10% ({len(extreme_negative)})')
    
    ax3.legend()
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('lstm_portfolio_analysis.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'lstm_portfolio_analysis.png'")
    
    # Print summary statistics
    print(f"\n=== LSTM Portfolio Analysis ===")
    print(f"Period: {df['date'].iloc[0].strftime('%Y-%m-%d')} to {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"Total days: {len(df)}")
    print(f"Initial value: $100,000")
    print(f"Final value: ${df['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Total return: {df['cumulative_return'].iloc[-1]:.2%}")
    
    print(f"\nDaily Return Statistics:")
    print(f"Mean: {df['return'].mean():.4f} ({df['return'].mean()*100:.2f}%)")
    print(f"Std: {df['return'].std():.4f} ({df['return'].std()*100:.2f}%)")
    print(f"Min: {df['return'].min():.4f} ({df['return'].min()*100:.2f}%)")
    print(f"Max: {df['return'].max():.4f} ({df['return'].max()*100:.2f}%)")
    
    print(f"\nExtreme Returns:")
    print(f"Days with return > +10%: {len(extreme_positive)}")
    print(f"Days with return < -10%: {len(extreme_negative)}")
    
    # Find the worst drawdown
    running_max = df['portfolio_value'].expanding().max()
    drawdown = (df['portfolio_value'] - running_max) / running_max
    max_drawdown = drawdown.min()
    max_drawdown_date = df.loc[drawdown.idxmin(), 'date']
    
    print(f"\nWorst Drawdown: {max_drawdown:.2%} on {max_drawdown_date.strftime('%Y-%m-%d')}")
    
    # Show when portfolio value hits concerning levels
    low_thresholds = [50000, 25000, 10000, 0]
    for threshold in low_thresholds:
        low_days = df[df['portfolio_value'] < threshold]
        if len(low_days) > 0:
            first_low = low_days.iloc[0]
            print(f"Portfolio first dropped below ${threshold:,} on {first_low['date'].strftime('%Y-%m-%d')}")
    
    # Find the 5 worst daily returns
    worst_days = df.nsmallest(5, 'return')
    print(f"\n5 Worst Daily Returns:")
    for _, day in worst_days.iterrows():
        print(f"  {day['date'].strftime('%Y-%m-%d')}: {day['return']*100:.2f}% (Portfolio: ${day['portfolio_value']:,.0f})")
    
    plt.show()

if __name__ == "__main__":
    plot_lstm_returns()