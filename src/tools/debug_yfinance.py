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

if __name__ == "__main__":
    # Test with a few tickers
    tickers = ["AAPL"]
    
    for ticker in tickers:
        result = debug_yfinance_data(ticker)
        sharesinfo = debug_yfinance_shares(ticker)
        print("\n" + "="*50 + "\n")