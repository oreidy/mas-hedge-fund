import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
import requests
import zipfile
from io import BytesIO
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
    Downloads 3-month Treasury yield from FRED and converts to daily risk-free rate.
    """
    # DGS3MO is the FRED series ID for 3-Month Treasury Constant Maturity Rate, Daily, Investment Basis
    t_bill_yields = get_fred_data(series_id="DGS3MO", start_date=start_date, end_date=end_date)
    
    if t_bill_yields.empty:
        print("Could not download T-bill yields from FRED. Please check FRED_API_KEY and internet connection.")
        return pd.Series()

    # FRED data is in percentage points (e.g., 4.42 for 4.42%). Convert to decimal.
    t_bill_yields['value'] = t_bill_yields['value'] / 100

    # Convert annualized yield to daily rate
    # The formula for daily rate from annualized rate is (1 + annual_rate)^(1/252) - 1
    daily_risk_free_rate = (1 + t_bill_yields['value'])**(1/252) - 1
    daily_risk_free_rate.name = 'risk_free_rate'
    return daily_risk_free_rate

def download_fama_french_factors(factor_type="3factor", start_date="2022-01-01", end_date="2025-06-30"):
    """
    Downloads Fama-French factor data from Kenneth French's data library.
    
    Args:
        factor_type: "3factor" or "5factor"
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        DataFrame with factor returns
    """
    
    if factor_type == "3factor":
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        csv_name = "F-F_Research_Data_Factors_daily.csv"
    elif factor_type == "5factor":
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        csv_name = "F-F_Research_Data_5_Factors_2x3_daily.csv"
    else:
        raise ValueError("factor_type must be '3factor' or '5factor'")
    
    try:
        print(f"Downloading {factor_type} Fama-French data...")
        
        # Download and extract ZIP file
        response = requests.get(url)
        response.raise_for_status()
        
        with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
            # Extract CSV content
            csv_content = zip_file.read(csv_name).decode('utf-8')
        
        # Parse CSV content
        lines = csv_content.strip().split('\n')
        
        # Find the start of data (skip header rows)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and line[0].isdigit():
                data_start = i
                break
        
        # Find the end of data (before annual data section)
        data_end = len(lines)
        for i in range(data_start, len(lines)):
            if 'Annual' in lines[i] or len(lines[i].strip()) == 0:
                data_end = i
                break
        
        # Parse the data section
        data_lines = lines[data_start:data_end]
        
        data_rows = []
        for line in data_lines:
            if line.strip():
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 4 and parts[0].isdigit():
                    data_rows.append(parts)
        
        if not data_rows:
            print(f"No data found in {factor_type} file")
            return pd.DataFrame()
        
        # Create DataFrame
        if factor_type == "3factor":
            columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']
        else:
            columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        
        df = pd.DataFrame(data_rows, columns=columns[:len(data_rows[0])])
        
        # Convert date column (YYYYMMDD format)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
        
        # Convert numeric columns and handle missing values
        numeric_cols = [col for col in df.columns if col != 'Date']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Replace -99 or similar missing value codes with NaN
            df[col] = df[col].replace([-99, -999, -99.99], np.nan)
            # Convert from percentage to decimal
            df[col] = df[col] / 100.0
        
        # Set date as index
        df.set_index('Date', inplace=True)
        
        # Filter by date range
        start_date_parsed = pd.to_datetime(start_date)
        end_date_parsed = pd.to_datetime(end_date)
        df = df[(df.index >= start_date_parsed) & (df.index <= end_date_parsed)]
        
        print(f"Downloaded {len(df)} {factor_type} factor observations from {df.index.min()} to {df.index.max()}")
        return df
        
    except Exception as e:
        print(f"Error downloading {factor_type} Fama-French data: {e}")
        return pd.DataFrame()

def main():
    """
    Main function to download benchmark data and calculate returns.
    """
    # Create directory to store benchmark data
    if not os.path.exists("benchmarking/data"):
        os.makedirs("benchmarking/data")

    # Tickers for benchmarks
    tickers = ["SPY", "HDG", "AGG"]  # SPY=stocks, HDG=hedge fund benchmark, AGG=bonds

    # Download and calculate returns
    all_returns = download_and_calculate_returns(tickers)

    # Save individual returns
    all_returns['SPY'].to_csv("benchmarking/data/spy_returns.csv")
    all_returns['HDG'].to_csv("benchmarking/data/hdg_returns.csv") 
    all_returns['AGG'].to_csv("benchmarking/data/agg_returns.csv")
    print("SPY, HDG, and AGG returns saved to benchmarking/data/")

    # Calculate and save 60/40 portfolio returns (60% stocks, 40% bonds)
    portfolio_returns = 0.6 * all_returns['SPY'] + 0.4 * all_returns['AGG']
    portfolio_returns.to_csv("benchmarking/data/60_40_portfolio_returns.csv")
    print("60/40 portfolio returns (60% SPY, 40% AGG) saved to benchmarking/data/60_40_portfolio_returns.csv")

    # Download and save risk-free rate
    risk_free_rate = download_risk_free_rate()
    if not risk_free_rate.empty:
        risk_free_rate.to_csv("benchmarking/data/daily_risk_free_rate.csv")
        print("Daily risk-free rate saved to benchmarking/data/daily_risk_free_rate.csv")

    # Download and save Fama-French factors
    ff_3factor = download_fama_french_factors("3factor")
    if not ff_3factor.empty:
        ff_3factor.to_csv("benchmarking/data/ff_3factor_daily.csv")
        print("Fama-French 3-factor data saved to benchmarking/data/ff_3factor_daily.csv")
    
    ff_5factor = download_fama_french_factors("5factor")
    if not ff_5factor.empty:
        ff_5factor.to_csv("benchmarking/data/ff_5factor_daily.csv")
        print("Fama-French 5-factor data saved to benchmarking/data/ff_5factor_daily.csv")

if __name__ == "__main__":
    main()
