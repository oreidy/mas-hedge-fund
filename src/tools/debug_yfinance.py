import yfinance as yf
import pandas as pd
import json
from pprint import pprint

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
        for key in ['longBusinessSummary', 'sector', 'industry', 
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

if __name__ == "__main__":
    # Test with a few tickers
    tickers = ["AAPL", "TSLA", "MSFT"]
    
    for ticker in tickers:
        result = debug_yfinance_data(ticker)
        print("\n" + "="*50 + "\n")