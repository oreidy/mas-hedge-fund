import requests
import aiohttp
import asyncio
import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import time
import random
from pathlib import Path
import json
from typing import List, Dict, Optional, Any

from data.models import InsiderTrade

# Constants for SEC EDGAR
SEC_EDGAR_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"

# Create a cache directory for SEC data
SEC_CACHE_DIR = Path("./cache/sec")
SEC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Headers to make requests look like they're coming from a browser
# This is important for SEC.gov which blocks requests without proper headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

async def fetch_insider_trades_for_ticker(ticker: str, start_date: str = None, end_date: str = None) -> List[InsiderTrade]:
    """
    Scrape insider trades for a specific ticker from SEC EDGAR.
    
    Args:
        ticker: The stock ticker symbol
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        
    Returns:
        List of InsiderTrade objects
    """
    cache_file = SEC_CACHE_DIR / f"{ticker}_insider_trades.json"
    
    # Try to load from cache first
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            # Filter by date if needed
            filtered_data = []
            for trade in cached_data:
                trade_date = trade.get("transaction_date") or trade.get("filing_date")
                if (not start_date or trade_date >= start_date) and (not end_date or trade_date <= end_date):
                    filtered_data.append(InsiderTrade(**trade))
                    
            # If we have data and it's recent (within last 7 days), return it
            if filtered_data and (datetime.now() - datetime.strptime(cached_data[-1]["filing_date"], "%Y-%m-%d")).days < 7:
                return filtered_data
        except Exception as e:
            print(f"Error loading cached insider trades for {ticker}: {e}")
    
    # Form 4 is for insider trading reports
    form_type = "4"
    
    # Calculate date range if not provided
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)
        start_date = start_date_dt.strftime("%Y-%m-%d")
    
    # Format dates for SEC EDGAR
    start_date_sec = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
    end_date_sec = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
    
    insider_trades = []
    
    try:
        # First, get the CIK number for the company
        cik = await get_cik_number(ticker)
        if not cik:
            print(f"Could not find CIK for {ticker}")
            return []
        
        # Make a request to SEC EDGAR to get Form 4 filings
        async with aiohttp.ClientSession() as session:
            params = {
                "action": "getcompany",
                "CIK": cik,
                "type": form_type,
                "dateb": end_date_sec,
                "datea": start_date_sec,
                "owner": "include",
                "output": "xml",
                "count": "100"
            }
            
            # Add a delay to avoid hitting SEC rate limits
            await asyncio.sleep(0.1 + random.random() * 0.2)
            
            async with session.get(SEC_EDGAR_URL, params=params, headers=HEADERS) as response:
                if response.status != 200:
                    print(f"Error fetching insider trades for {ticker}: Status {response.status}")
                    return []
                
                text = await response.text()
                
                # Parse the XML to find Form 4 links
                soup = BeautifulSoup(text, 'html.parser')
                filing_entries = soup.find_all('entry')
                
                for entry in filing_entries:
                    filing_href = entry.find('filing-href')
                    if not filing_href:
                        continue
                        
                    filing_url = filing_href.text
                    
                    # Get Form 4 data
                    trades = await parse_form4(session, filing_url, ticker)
                    insider_trades.extend(trades)
                    
                    # Add a delay between requests
                    await asyncio.sleep(0.1 + random.random() * 0.2)
        
        # Sort by transaction date
        insider_trades.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        
        # Cache the results
        if insider_trades:
            with open(cache_file, 'w') as f:
                json.dump([trade.model_dump() for trade in insider_trades], f)
        
        return insider_trades
    
    except Exception as e:
        print(f"Error fetching insider trades for {ticker}: {e}")
        return []

async def get_cik_number(ticker: str) -> Optional[str]:
    """
    Get the CIK (Central Index Key) for a ticker from the SEC.
    
    Args:
        ticker: The stock ticker symbol
        
    Returns:
        CIK number as string or None if not found
    """
    # Check cache first
    cache_file = SEC_CACHE_DIR / "cik_lookup.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cik_lookup = json.load(f)
                if ticker in cik_lookup:
                    return cik_lookup[ticker]
        except Exception:
            pass
    
    # Use SEC's ticker to CIK JSON API
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://www.sec.gov/files/company_tickers.json"
            
            async with session.get(url, headers=HEADERS) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                
                # Search for the ticker in the data
                for _, company in data.items():
                    if company["ticker"].upper() == ticker.upper():
                        cik = str(company["cik_str"]).zfill(10)
                        
                        # Save to cache
                        cik_lookup = {}
                        if cache_file.exists():
                            with open(cache_file, 'r') as f:
                                cik_lookup = json.load(f)
                        
                        cik_lookup[ticker] = cik
                        with open(cache_file, 'w') as f:
                            json.dump(cik_lookup, f)
                        
                        return cik
    except Exception as e:
        print(f"Error getting CIK for {ticker}: {e}")
    
    return None

async def parse_form4(session: aiohttp.ClientSession, form_url: str, ticker: str) -> List[InsiderTrade]:
    """
    Parse a Form 4 filing to extract insider trading information.
    
    Args:
        session: aiohttp session
        form_url: URL to the Form 4 filing
        ticker: Ticker symbol for the company
        
    Returns:
        List of InsiderTrade objects
    """
    try:
        # First, get the Form 4 page
        async with session.get(form_url, headers=HEADERS) as response:
            if response.status != 200:
                return []
            
            text = await response.text()
            
        # Extract the link to the actual XML data file
        soup = BeautifulSoup(text, 'html.parser')
        xml_link = None
        for link in soup.find_all('a'):
            if link.get('href', '').endswith('.xml'):
                xml_link = 'https://www.sec.gov' + link['href']
                break
        
        if not xml_link:
            return []
        
        # Get the XML data
        async with session.get(xml_link, headers=HEADERS) as response:
            if response.status != 200:
                return []
            
            xml_text = await response.text()
        
        soup = BeautifulSoup(xml_text, 'xml')
        
        # Extract information about the issuer
        issuer_element = soup.find('issuer')
        if not issuer_element:
            return []
            
        issuer_name = issuer_element.find('issuerName').text if issuer_element.find('issuerName') else None
        issuer_ticker = issuer_element.find('issuerTradingSymbol').text if issuer_element.find('issuerTradingSymbol') else None
        
        # If the ticker doesn't match, skip this form
        if issuer_ticker and issuer_ticker.upper() != ticker.upper():
            return []
        
        # Extract information about the reporting owner
        owner_element = soup.find('reportingOwner')
        owner_name = owner_element.find('rptOwnerName').text if owner_element and owner_element.find('rptOwnerName') else None
        
        # Check if owner is a director
        is_director = False
        relationship = owner_element.find('reportingOwnerRelationship') if owner_element else None
        if relationship and relationship.find('isDirector') and relationship.find('isDirector').text.upper() == 'TRUE':
            is_director = True
        
        # Get owner's title
        title = None
        if relationship and relationship.find('officerTitle'):
            title = relationship.find('officerTitle').text
        
        # Extract filing date
        filing_date_elem = soup.find('periodOfReport')
        filing_date = filing_date_elem.text if filing_date_elem else None
        if filing_date:
            filing_date = datetime.strptime(filing_date, '%Y-%m-%d').strftime('%Y-%m-%d')
        
        # Extract transactions
        transactions = []
        non_derivative_table = soup.find('nonDerivativeTable')
        
        if non_derivative_table:
            transaction_elements = non_derivative_table.find_all('nonDerivativeTransaction')
            
            for transaction in transaction_elements:
                # Extract transaction details
                transaction_date_elem = transaction.find('transactionDate').find('value') if transaction.find('transactionDate') else None
                transaction_date = transaction_date_elem.text if transaction_date_elem else None
                
                if transaction_date:
                    transaction_date = datetime.strptime(transaction_date, '%Y-%m-%d').strftime('%Y-%m-%d')
                
                # Security information
                security_title_elem = transaction.find('securityTitle').find('value') if transaction.find('securityTitle') else None
                security_title = security_title_elem.text if security_title_elem else None
                
                # Transaction amounts
                shares_elem = transaction.find('transactionAmounts').find('transactionShares').find('value') if transaction.find('transactionAmounts') else None
                price_elem = transaction.find('transactionAmounts').find('transactionPricePerShare').find('value') if transaction.find('transactionAmounts') else None
                
                shares = float(shares_elem.text) if shares_elem else None
                price = float(price_elem.text) if price_elem else None
                
                # Transaction code (P=Purchase, S=Sale, etc.)
                transaction_code_elem = transaction.find('transactionAmounts').find('transactionAcquiredDisposedCode').find('value') if transaction.find('transactionAmounts') else None
                transaction_code = transaction_code_elem.text if transaction_code_elem else None
                
                # If it's a sale, make shares negative
                if transaction_code == 'D':  # D = Disposed (sold)
                    shares = -shares if shares else None
                
                # Shares owned after transaction
                shares_owned_after_elem = transaction.find('postTransactionAmounts').find('sharesOwnedFollowingTransaction').find('value') if transaction.find('postTransactionAmounts') else None
                shares_owned_after = float(shares_owned_after_elem.text) if shares_owned_after_elem else None
                
                # Calculate shares owned before transaction
                shares_owned_before = shares_owned_after - shares if shares_owned_after is not None and shares is not None else None
                
                # Calculate transaction value
                transaction_value = shares * price if shares is not None and price is not None else None
                
                # Create InsiderTrade object
                trade = InsiderTrade(
                    ticker=ticker,
                    issuer=issuer_name,
                    name=owner_name,
                    title=title,
                    is_board_director=is_director,
                    transaction_date=transaction_date,
                    transaction_shares=shares,
                    transaction_price_per_share=price,
                    transaction_value=transaction_value,
                    shares_owned_before_transaction=shares_owned_before,
                    shares_owned_after_transaction=shares_owned_after,
                    security_title=security_title,
                    filing_date=filing_date or transaction_date
                )
                
                transactions.append(trade)
        
        return transactions
    
    except Exception as e:
        print(f"Error parsing Form 4 at {form_url}: {e}")
        return []

async def fetch_multiple_insider_trades(tickers: List[str], start_date: str = None, end_date: str = None) -> Dict[str, List[InsiderTrade]]:
    """
    Fetch insider trades for multiple tickers in parallel.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        
    Returns:
        Dictionary of ticker to insider trades list
    """
    tasks = [fetch_insider_trades_for_ticker(ticker, start_date, end_date) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    
    return {ticker: trades for ticker, trades in zip(tickers, results)}