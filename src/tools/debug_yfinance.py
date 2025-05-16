import yfinance as yf
import pandas as pd
import json
from pprint import pprint
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to sys.path to import from your project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the specific functions we want to debug
from tools.api import get_price_data, get_prices, fetch_prices_batch

def debug_yfinance_data(ticker_symbol):
    """
    Debug function to inspect what financial data yfinance is actually getting.
    """
    print(f"=== DEBUGGING YFINANCE FOR {ticker_symbol} ===")
    
    ticker = yf.Ticker(ticker_symbol)
    
    # Check if we can get any financial statements
    balance_sheet = ticker.balance_sheet
    income_stmt = ticker.income_stmt
    cashflow = ticker.cashflow
    
    # Print available financial statements
    print(f"\nBalance Sheet available: {not balance_sheet.empty}")
    if not balance_sheet.empty:
        print(f"Balance Sheet shape: {balance_sheet.shape}")
        print(f"Balance Sheet fields: {list(balance_sheet.index)}")
    
    print(f"\nIncome Statement available: {not income_stmt.empty}")
    if not income_stmt.empty:
        print(f"Income Statement shape: {income_stmt.shape}")
        print(f"Income Statement fields: {list(income_stmt.index)}")
    
    print(f"\nCash Flow available: {not cashflow.empty}")
    if not cashflow.empty:
        print(f"Cash Flow shape: {cashflow.shape}")
        print(f"Cash Flow fields: {list(cashflow.index)}")
    
    # Try quarterly statements
    print("\nChecking quarterly statements:")
    qbs = ticker.quarterly_balance_sheet
    qis = ticker.quarterly_income_stmt
    qcf = ticker.quarterly_cashflow
    
    print(f"Quarterly Balance Sheet available: {not qbs.empty}")
    print(f"Quarterly Income Statement available: {not qis.empty}")
    print(f"Quarterly Cash Flow available: {not qcf.empty}")
    
    # Try using .info() which might have more data
    print("\nTrying ticker.info:")
    try:
        info_keys = list(ticker.info.keys())
        print(f"Info available fields: {info_keys}")
        print("\nSample values:")
        for key in ['sector', 'industry', 
                    'marketCap', 'sharesOutstanding']:
            if key in ticker.info:
                print(f"{key}: {ticker.info[key]}")
    except Exception as e:
        print(f"Error getting info: {e}")
    
    # Try alternative approach
    print("\nTrying ticker.financials:")
    try:
        financials = ticker.financials
        print(f"Financials available: {not financials.empty}")
        if not financials.empty:
            print(f"Financials shape: {financials.shape}")
            print(f"Financials fields: {list(financials.index)}")
    except Exception as e:
        print(f"Error getting financials: {e}")
    
    return {
        "balance_sheet": balance_sheet,
        "income_stmt": income_stmt,
        "cashflow": cashflow,
        "info_available": hasattr(ticker, "info") and ticker.info is not None,
        "financials_available": not ticker.financials.empty if hasattr(ticker, "financials") else False
    }

def debug_yfinance_shares(ticker_symbol):
    """
    Debug function specifically focused on tracking outstanding shares over time.
    """
    print(f"=== DEBUGGING OUTSTANDING SHARES FOR {ticker_symbol} ===")
    
    ticker = yf.Ticker(ticker_symbol)
    
    # Get balance sheets (annual and quarterly)
    balance_sheet = ticker.balance_sheet
    qbs = ticker.quarterly_balance_sheet
    
    # Check for share count in annual balance sheet
    print("\n1. ANNUAL BALANCE SHEET APPROACH:")
    if not balance_sheet.empty:
        print(f"Annual Balance Sheet columns (dates): {[col.strftime('%Y-%m-%d') for col in balance_sheet.columns]}")
        
        # Look for fields that might contain share data
        share_related_fields = [
            "Ordinary Shares Number",
            "Share Issued",
            "Common Stock Shares Outstanding",
            "Shares Outstanding",
            "Common Stock"
        ]
        
        found = False
        for field in share_related_fields:
            if field in balance_sheet.index:
                print(f"\nFound field: {field}")
                print(f"Values by date:")
                for date in balance_sheet.columns:
                    print(f"  {date.strftime('%Y-%m-%d')}: {balance_sheet.loc[field, date]:,.0f}")
                found = True
        
        if not found:
            print("No share-related fields found in annual balance sheet.")
    else:
        print("Annual balance sheet data not available.")
    
    # Check for share count in quarterly balance sheet
    print("\n2. QUARTERLY BALANCE SHEET APPROACH:")
    if not qbs.empty:
        print(f"Quarterly Balance Sheet columns (dates): {[col.strftime('%Y-%m-%d') for col in qbs.columns]}")
        
        # Look for fields that might contain share data
        share_related_fields = [
            "Ordinary Shares Number",
            "Share Issued",
            "Common Stock Shares Outstanding",
            "Shares Outstanding",
            "Common Stock"
        ]
        
        found = False
        for field in share_related_fields:
            if field in qbs.index:
                print(f"\nFound field: {field}")
                print(f"Values by date:")
                for date in qbs.columns:
                    print(f"  {date.strftime('%Y-%m-%d')}: {qbs.loc[field, date]:,.0f}")
                found = True
        
        if not found:
            print("No share-related fields found in quarterly balance sheet.")
    else:
        print("Quarterly balance sheet data not available.")
    
    # Check info object for current shares outstanding
    print("\n3. INFO OBJECT APPROACH:")
    try:
        if 'sharesOutstanding' in ticker.info:
            print(f"Current shares outstanding (from info): {ticker.info['sharesOutstanding']:,.0f}")
        else:
            print("'sharesOutstanding' not found in info object.")
            
        if 'floatShares' in ticker.info:
            print(f"Float shares (from info): {ticker.info['floatShares']:,.0f}")
    except Exception as e:
        print(f"Error getting share info: {e}")
    
    # Try historical data approach
    print("\n4. HISTORICAL DATA APPROACH:")
    try:
        # Try to find historical share data in key stats or other endpoints
        key_stats = ticker.stats()
        if 'sharesOutstanding' in key_stats:
            print(f"Shares outstanding from key stats: {key_stats['sharesOutstanding']:,.0f}")
    except Exception as e:
        print(f"Error getting historical share data: {e}")
    
    # Try SEC filings approach
    print("\n5. SEC FILINGS APPROACH:")
    try:
        # Get recent filings - this is a stretch as yfinance doesn't easily provide this
        recent_actions = ticker.get_recent_actions()
        print(f"Recent actions available: {not recent_actions.empty}")
        if not recent_actions.empty:
            print(recent_actions.head())
    except Exception as e:
        print(f"Error getting SEC filings: {e}")
    
    # Alternative sources
    print("\n6. CHECKING DIRECT API DATA:")
    try:
        # Direct query of data endpoints
        info = ticker.info
        if 'marketCap' in info and 'currentPrice' in info:
            market_cap = info['marketCap']
            current_price = info['currentPrice']
            estimated_shares = market_cap / current_price
            print(f"Estimated shares from market cap / price: {estimated_shares:,.0f}")
    except Exception as e:
        print(f"Error calculating estimated shares: {e}")
    
    # Get historical share buybacks data if available
    print("\n7. HISTORICAL SHARE BUYBACKS:")
    try:
        # This is speculative as yfinance doesn't have a direct method for this
        cashflow = ticker.cashflow
        if not cashflow.empty and "Repurchase Of Capital Stock" in cashflow.index:
            print("Share buyback data from cash flow statement:")
            for date in cashflow.columns:
                value = cashflow.loc["Repurchase Of Capital Stock", date]
                print(f"  {date.strftime('%Y-%m-%d')}: ${value:,.0f}")
    except Exception as e:
        print(f"Error getting buyback data: {e}")
    
    return {
        "ticker": ticker_symbol,
        "balance_sheet_dates": [col.strftime('%Y-%m-%d') for col in balance_sheet.columns] if not balance_sheet.empty else [],
        "current_shares": ticker.info.get('sharesOutstanding', None)
    }

def debug_price_data_fetching(ticker_symbol, start_date=None, end_date=None):
    """
    Debug function specifically to test the get_price_data function and diagnose issues
    with fetching multiple days of price data.
    """
    print(f"=== DEBUGGING PRICE DATA FETCHING FOR {ticker_symbol} ===")
    
    # Use default dates if not provided
    if not start_date:
        # Default to 7 days ago
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    if not end_date:
        # Default to today
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Debugging date range: {start_date} to {end_date}")
    
    # Test 1: Direct yfinance download
    print("\n1. TESTING DIRECT YFINANCE DOWNLOAD:")
    try:
        # Download with yfinance directly
        yf_data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
        print(f"yfinance direct download result:")
        print(f"  - Shape: {yf_data.shape}")
        print(f"  - Date range: {yf_data.index.min()} to {yf_data.index.max()}")
        print(f"  - Number of trading days: {len(yf_data)}")
        print(f"  - First few rows:")
        print(yf_data.head())
    except Exception as e:
        print(f"Error with direct yfinance download: {e}")
    
    # Test 2: Using your get_price_data function
    print("\n2. TESTING YOUR get_price_data FUNCTION:")
    try:
        price_df = get_price_data(ticker_symbol, start_date, end_date, verbose_data=True)
        print(f"get_price_data result:")
        print(f"  - Shape: {price_df.shape}")
        print(f"  - Date range: {price_df.index.min()} to {price_df.index.max()}")
        print(f"  - Number of trading days: {len(price_df)}")
        print(f"  - First few rows:")
        print(price_df.head())
        print(f"  - Last few rows:")
        print(price_df.tail())
    except Exception as e:
        print(f"Error with get_price_data: {e}")
    
    # Test 3: Try with a single day range to replicate the issue
    single_day_start = end_date
    single_day_end = end_date
    print(f"\n3. TESTING WITH SINGLE DAY RANGE ({single_day_start} to {single_day_end}):")
    try:
        # Direct yfinance
        yf_single_data = yf.download(ticker_symbol, start=single_day_start, end=single_day_end, progress=False)
        print(f"yfinance direct download for single day:")
        print(f"  - Shape: {yf_single_data.shape}")
        print(f"  - Number of rows: {len(yf_single_data)}")
        
        # Your function
        try:
            single_df = get_price_data(ticker_symbol, single_day_start, single_day_end, verbose_data=True)
            print(f"get_price_data for single day:")
            print(f"  - Shape: {single_df.shape}")
            print(f"  - Number of rows: {len(single_df)}")
        except Exception as e:
            print(f"Error with get_price_data for single day: {e}")
    except Exception as e:
        print(f"Error with direct yfinance download for single day: {e}")
    
    # Test 4: Test with a two-day range
    two_day_start = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    two_day_end = end_date
    print(f"\n4. TESTING WITH TWO DAY RANGE ({two_day_start} to {two_day_end}):")
    try:
        # Direct yfinance
        yf_two_data = yf.download(ticker_symbol, start=two_day_start, end=two_day_end, progress=False)
        print(f"yfinance direct download for two days:")
        print(f"  - Shape: {yf_two_data.shape}")
        print(f"  - Number of rows: {len(yf_two_data)}")
        
        # Your function
        try:
            two_df = get_price_data(ticker_symbol, two_day_start, two_day_end, verbose_data=True)
            print(f"get_price_data for two days:")
            print(f"  - Shape: {two_df.shape}")
            print(f"  - Number of rows: {len(two_df)}")
        except Exception as e:
            print(f"Error with get_price_data for two days: {e}")
    except Exception as e:
        print(f"Error with direct yfinance download for two days: {e}")
    
    # Test 5: Extended range test
    extended_start = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
    extended_end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\n5. TESTING WITH EXTENDED RANGE ({extended_start} to {extended_end}):")
    try:
        yf_extended = yf.download(ticker_symbol, start=extended_start, end=extended_end, progress=False)
        print(f"yfinance direct download with extended range:")
        print(f"  - Shape: {yf_extended.shape}")
        print(f"  - Number of rows: {len(yf_extended)}")
        print(f"  - Date range: {yf_extended.index.min()} to {yf_extended.index.max()}")
    except Exception as e:
        print(f"Error with extended range test: {e}")

def print_financial_statements(ticker_symbol, end_date="2023-03-12", periods=2):
    """
    Simple function to print raw financial statements for manual cross-checking.
    
    Args:
        ticker_symbol: The stock ticker symbol
        end_date: Optional end date (YYYY-MM-DD) to match your backtest
        periods: Number of periods to display (default: 2)
    """
    print(f"\n=== FINANCIAL STATEMENTS FOR {ticker_symbol} ===")
    if end_date:
        print(f"For date: {end_date}")
    
    ticker = yf.Ticker(ticker_symbol)
    
    # Get financial statements
    balance_sheet = ticker.balance_sheet
    income_stmt = ticker.income_stmt  
    cash_flow = ticker.cashflow
    
    # Filter by end_date if provided
    if end_date and not balance_sheet.empty:
        # Convert columns to string format for comparison
        date_strs = [col.strftime('%Y-%m-%d') for col in balance_sheet.columns]
        # Keep only columns where date <= end_date
        valid_cols = [i for i, date_str in enumerate(date_strs) if date_str <= end_date]
        if valid_cols:
            balance_sheet = balance_sheet.iloc[:, valid_cols]
    
    if end_date and not income_stmt.empty:
        # Convert columns to string format for comparison
        date_strs = [col.strftime('%Y-%m-%d') for col in income_stmt.columns]
        # Keep only columns where date <= end_date
        valid_cols = [i for i, date_str in enumerate(date_strs) if date_str <= end_date]
        if valid_cols:
            income_stmt = income_stmt.iloc[:, valid_cols]
    
    if end_date and not cash_flow.empty:
        # Convert columns to string format for comparison
        date_strs = [col.strftime('%Y-%m-%d') for col in cash_flow.columns]
        # Keep only columns where date <= end_date
        valid_cols = [i for i, date_str in enumerate(date_strs) if date_str <= end_date]
        if valid_cols:
            cash_flow = cash_flow.iloc[:, valid_cols]
    
    # Limit to requested number of periods
    if not balance_sheet.empty:
        balance_sheet = balance_sheet.iloc[:, :periods]
    if not income_stmt.empty:
        income_stmt = income_stmt.iloc[:, :periods]  
    if not cash_flow.empty:
        cash_flow = cash_flow.iloc[:, :periods]
    
    # Print Balance Sheet
    print("\n=== BALANCE SHEET ===")
    if not balance_sheet.empty:
        for col in balance_sheet.columns:
            print(f"Date: {col.strftime('%Y-%m-%d')}")
        
        # Print key items we're interested in
        important_items = [
            'Current Assets', 
            'Current Liabilities', 
            'Total Assets',
            'Total Liabilities Net Minority Interest',
            'Stockholders Equity',
            'Ordinary Shares Number'
        ]
        
        print("\nKey Balance Sheet Items:")
        for item in important_items:
            if item in balance_sheet.index:
                values = []
                for col in balance_sheet.columns:
                    values.append(f"${float(balance_sheet.loc[item, col]):,.2f}")
                print(f"{item}: {' | '.join(values)}")
            else:
                print(f"{item}: Not available")
        
        print("\nAll Balance Sheet Items:")
        for item in balance_sheet.index:
            try:
                values = []
                for col in balance_sheet.columns:
                    values.append(f"${float(balance_sheet.loc[item, col]):,.2f}")
                print(f"{item}: {' | '.join(values)}")
            except:
                print(f"{item}: Could not format values")
    else:
        print("No balance sheet data available")
    
    # Print Income Statement
    print("\n=== INCOME STATEMENT ===")
    if not income_stmt.empty:
        for col in income_stmt.columns:
            print(f"Date: {col.strftime('%Y-%m-%d')}")
        
        # Print key items we're interested in
        important_items = [
            'Total Revenue', 
            'Operating Income', 
            'Net Income'
        ]
        
        print("\nKey Income Statement Items:")
        for item in important_items:
            if item in income_stmt.index:
                values = []
                for col in income_stmt.columns:
                    values.append(f"${float(income_stmt.loc[item, col]):,.2f}")
                print(f"{item}: {' | '.join(values)}")
            else:
                print(f"{item}: Not available")
        
        print("\nAll Income Statement Items:")
        for item in income_stmt.index:
            try:
                values = []
                for col in income_stmt.columns:
                    values.append(f"${float(income_stmt.loc[item, col]):,.2f}")
                print(f"{item}: {' | '.join(values)}")
            except:
                print(f"{item}: Could not format values")
    else:
        print("No income statement data available")
    
    # Print Cash Flow Statement
    print("\n=== CASH FLOW STATEMENT ===")
    if not cash_flow.empty:
        for col in cash_flow.columns:
            print(f"Date: {col.strftime('%Y-%m-%d')}")
        
        # Print key items we're interested in
        important_items = [
            'Operating Cash Flow', 
            'Capital Expenditure', 
            'Free Cash Flow'
        ]
        
        print("\nKey Cash Flow Items:")
        for item in important_items:
            if item in cash_flow.index:
                values = []
                for col in cash_flow.columns:
                    values.append(f"${float(cash_flow.loc[item, col]):,.2f}")
                print(f"{item}: {' | '.join(values)}")
            else:
                print(f"{item}: Not available")
        
        print("\nAll Cash Flow Items:")
        for item in cash_flow.index:
            try:
                values = []
                for col in cash_flow.columns:
                    values.append(f"${float(cash_flow.loc[item, col]):,.2f}")
                print(f"{item}: {' | '.join(values)}")
            except:
                print(f"{item}: Could not format values")
    else:
        print("No cash flow statement data available")



if __name__ == "__main__":
    # Test with a few tickers
    tickers = ["WMT"]

    # For historical dates where we know there should be data
    start_date = "2023-03-10"  # This matches your backtest example issue
    end_date = "2023-03-14"    # This matches your backtest example issue
    
    for ticker in tickers:
        #result = debug_yfinance_data(ticker)
        #sharesinfo = debug_yfinance_shares(ticker)
        #debug_price_data_fetching(ticker, start_date, end_date)
        print_financial_statements(ticker, end_date, periods=2)

        print("\n" + "="*50 + "\n")