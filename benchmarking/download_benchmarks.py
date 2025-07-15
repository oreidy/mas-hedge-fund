import yfinance as yf
import pandas as pd
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tools.fred_api import get_fred_data

def download_and_calculate_returns(tickers, start_date="2022-01-01", end_date="2025-06-30"):
    """
    Downloads historical data for given tickers and calculates daily returns.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    returns = data.pct_change().dropna()
    return returns

def download_risk_free_rate(start_date="2022-01-01", end_date="2025-06-30"):
    """
    Downloads 3-month T-bill yield from FRED and converts to daily risk-free rate.
    """
    # TB3MS is the FRED series ID for 3-Month Treasury Bill Secondary Market Rate, Discount Basis
    t_bill_yields = get_fred_data(series_id="TB3MS", start_date=start_date, end_date=end_date)
    
    if t_bill_yields.empty:
        print("Could not download T-bill yields from FRED. Please check FRED_API_KEY and internet connection.")
        return pd.Series()

    # FRED data is typically in percentage points (e.g., 1.0 for 1%). Convert to decimal.
    t_bill_yields['value'] = t_bill_yields['value'] / 100

    # Convert annualized yield to daily rate
    # The formula for daily rate from annualized rate is (1 + annual_rate)^(1/252) - 1
    daily_risk_free_rate = (1 + t_bill_yields['value'])**(1/252) - 1
    daily_risk_free_rate.name = 'risk_free_rate'
    return daily_risk_free_rate

def main():
    """
    Main function to download benchmark data and calculate returns.
    """
    # Create directory to store benchmark data
    if not os.path.exists("benchmarking/data"):
        os.makedirs("benchmarking/data")

    # Tickers for benchmarks
    tickers = ["SPY", "HDG"]

    # Download and calculate returns
    all_returns = download_and_calculate_returns(tickers)

    # Save individual returns
    all_returns['SPY'].to_csv("benchmarking/data/spy_returns.csv")
    all_returns['HDG'].to_csv("benchmarking/data/hdg_returns.csv")
    print("SPY and HDG returns saved to benchmarking/data/")

    # Calculate and save 60/40 portfolio returns
    portfolio_returns = 0.6 * all_returns['SPY'] + 0.4 * all_returns['HDG']
    portfolio_returns.to_csv("benchmarking/data/60_40_portfolio_returns.csv")
    print("60/40 portfolio returns saved to benchmarking/data/60_40_portfolio_returns.csv")

    # Download and save risk-free rate
    risk_free_rate = download_risk_free_rate()
    if not risk_free_rate.empty:
        risk_free_rate.to_csv("benchmarking/data/daily_risk_free_rate.csv")
        print("Daily risk-free rate saved to benchmarking/data/daily_risk_free_rate.csv")

if __name__ == "__main__":
    main()
