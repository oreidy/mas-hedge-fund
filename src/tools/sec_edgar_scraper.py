import os
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import List, Dict, Optional
from pathlib import Path
import tempfile
from secedgar import filings, FilingType
import xml.etree.ElementTree as ET
import glob
import traceback

from utils.logger import logger
from data.models import InsiderTrade

from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Create a cache directory for SEC data
SEC_CACHE_DIR = Path(os.path.expanduser("~/ai-hedge-fund/src/tools/cache/sec_edgar_filings"))
SEC_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_multiple_insider_trades(tickers: List[str], end_date: str = None, start_date: str = None, verbose_data: bool = False) -> Dict[str, List[InsiderTrade]]:
    """
    Fetch insider trades for multiple tickers using the secedgar library.
    
    Args:
        tickers: List of stock ticker symbols
        end_date: Optional end date (YYYY-MM-DD)
        
    Returns:
        Dictionary mapping tickers to lists of InsiderTrade objects
    """
    if verbose_data:
        logger.debug(f"End date: {end_date}", module="sec_edgar_scraper")

    # Ensure tickers is a list of strings
    if isinstance(tickers, str):
        tickers = [tickers]

    # Calculate date range if not provided
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Default start_date to 10 days before end_date if not provided
    if not start_date:
        start_date_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=10)
        start_date = start_date_dt.strftime("%Y-%m-%d")
    
    # Convert string dates to datetime objects
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    results = {}
    
    # Check cache for each ticker
    cache_hits = 0
    tickers_to_fetch = []
    
    for ticker in tickers:
        cache_file = SEC_CACHE_DIR / f"{ticker}_insider_trades.json"

        #if verbose_data:
            #logger.debug(f"Checking cache at {cache_file}", module="sec_edgar_scraper", ticker=ticker)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                if verbose_data:
                    logger.debug(f"Found cache file with {len(cached_data)} entries", module="sec_edgar_scraper", ticker=ticker)
                
                # Filter by date range
                filtered_data = []
                for trade in cached_data:
                    trade_date = trade.get("transaction_date") or trade.get("filing_date")
                    if trade_date and start_date <= trade_date <= end_date:
                        filtered_data.append(InsiderTrade(**trade))
                
                if filtered_data:
                    if verbose_data:
                        logger.debug(f"Found {len(filtered_data)} trades in cache matching date range", module="sec_edgar_scraper", ticker=ticker)
                    
                    results[ticker] = filtered_data
                    cache_hits += 1
                    continue
                elif verbose_data:
                    logger.debug(f"No trades in cache match the date range", module="sec_edgar_scraper", ticker=ticker)
            except Exception as e:
                logger.error(f"Error loading cached insider trades for {ticker}: {e}", module="sec_edgar_scraper", ticker=ticker)
        elif verbose_data:
            logger.debug(f"No cache file found for {ticker}", module="sec_edgar_scraper", ticker=ticker)
        
        tickers_to_fetch.append(ticker)
    
    if not tickers_to_fetch:
        if verbose_data:
            logger.debug(f"All tickers found in cache, no need to fetch", module="sec_edgar_scraper")
            for ticker in tickers:
                if ticker in results and results[ticker]:
                    most_recent_date = max(trade.transaction_date or trade.filing_date for trade in results[ticker])
                    logger.debug(f"Most recent cached trade: {most_recent_date}", module="sec_edgar_scraper", ticker=ticker)
        return results

    
    # Use a temporary directory for Form 4 downloads.
    # # We don't need them after saving them as json in the cache
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            if verbose_data:
                logger.debug(f"Using temporary directory: {temp_dir}", module="sec_edgar_scraper")

            logger.debug(f"Fetching SEC filings for tickers: {tickers_to_fetch}, start: {start_dt}, end: {end_dt}", 
            module="sec_edgar_scraper")
            
            # Create a Filing object for Form 4
            my_filings = filings(
                cik_lookup=tickers_to_fetch,
                filing_type=FilingType.FILING_4,
                user_agent="Oliver Reidy (oliver.reidy@uzh.ch)",
                start_date=start_dt,
                end_date=end_dt,
            )
            
            if verbose_data:
                logger.debug(f"Created filing object: {type(my_filings)}", module="sec_edgar_scraper")

            # Download the filings
            my_filings.save(directory=str(temp_dir))
            if verbose_data:
                logger.debug(f"Saved filings to {temp_dir}", module="sec_edgar_scraper")

            # Debug: List all files in the temp directory to see the structure
            if verbose_data:
                logger.debug("Directory structure:", module="sec_edgar_scraper")
                for root, dirs, files in os.walk(temp_dir):
                    logger.debug(f"Root: {root}", module="sec_edgar_scraper")
                    logger.debug(f"Dirs: {dirs}", module="sec_edgar_scraper")
                    logger.debug(f"Files: {len(files)}", module="sec_edgar_scraper")
            
            # Process the downloaded files for each ticker
            for ticker in tickers_to_fetch:
                insider_trades = []
                
                ticker_dir = os.path.join(temp_dir, ticker)
                form4_dir = os.path.join(ticker_dir, "4")
                
                if os.path.exists(form4_dir):
                    #if verbose_data:
                        #logger.debug(f"Found Form 4 directory: {form4_dir}", module="sec_edgar_scraper", ticker=ticker)
                    
                    # Process all TXT files in the directory
                    for txt_file in glob.glob(os.path.join(form4_dir, "**/*.txt"), recursive=True):
                        #if verbose_data:
                            #logger.debug(f"Processing TXT file: {txt_file}", module="sec_edgar_scraper", ticker=ticker)
                        trades = parse_form4_file(txt_file, ticker)
                        insider_trades.extend(trades)
                else:
                    logger.debug(f"No downloaded insider trades found", module="sec_edgar_scraper", ticker=ticker)
                
                # Sort by transaction date
                insider_trades.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
                
                # Cache the results
                if insider_trades:
                    cache_file = SEC_CACHE_DIR / f"{ticker}_insider_trades.json"
                    with open(cache_file, 'w') as f:
                        json.dump([trade.model_dump() for trade in insider_trades], f)
                    logger.debug(f"Cached {len(insider_trades)} insider trades", module="sec_edgar_scraper", ticker=ticker)
                else:
                    logger.warning(f"No insider trades to cache", module="sec_edgar_scraper", ticker=ticker)
                
                results[ticker] = insider_trades
        
        except Exception as e:
            if "No filings available" in str(e):
                logger.info(f"No insider trading filings found for the requested date range", 
                        module="sec_edgar_scraper")
            else:
                logger.warning(f"Error fetching insider trades: {e}. Activate verbose_data for detailed error", 
                            module="sec_edgar_scraper")
            
            if verbose_data:
                # Print more detailed error information
                logger.error(f"Error details: {traceback.format_exc()}", 
                            module="sec_edgar_scraper")
    
            # For any tickers that failed, return empty lists
            for ticker in tickers_to_fetch:
                if ticker not in results:
                    results[ticker] = []
        
        # Debug:
        if verbose_data:
            for ticker in tickers_to_fetch:
                if ticker in results and results[ticker]:
                    most_recent_date = max(trade.transaction_date or trade.filing_date for trade in results[ticker])
                    logger.debug(f"Most recent trade: {most_recent_date}", module="sec_edgar_scraper", ticker=ticker)
        
        return results

def parse_form4_file(file_path: str, ticker: str) -> List[InsiderTrade]:
    """
    Parse a Form 4 XML file to extract insider trading information.
    
    Args:
        file_path: Path to the Form 4 XML file
        ticker: Ticker symbol for the company
        
    Returns:
        List of InsiderTrade objects
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract the XML portion from the SEC EDGAR file
        xml_start = content.find('<XML>')
        xml_end = content.find('</XML>')

        if xml_start == -1 or xml_end == -1:
            logger.debug(f"No XML tags found in {file_path}", module="sec_edgar_scraper")
            return []
        
        # Extract the XML content (add 5 to skip the <XML> tag itself)
        xml_content = content[xml_start + 5:xml_end].strip()

        # Create a file-like object from the XML content
        import io
        xml_file = io.StringIO(xml_content)

        # Parse the txt file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extract namespace from root tag if present
        ns = ''
        if '}' in root.tag:
            ns = root.tag.split('}')[0] + '}'
        
        # Extract information about the issuer
        issuer_element = root.find(f'.//{ns}issuer')
        if issuer_element is None:
            return []
            
        issuer_name = issuer_element.find(f'.//{ns}issuerName').text if issuer_element.find(f'.//{ns}issuerName') is not None else None
        issuer_ticker = issuer_element.find(f'.//{ns}issuerTradingSymbol').text if issuer_element.find(f'.//{ns}issuerTradingSymbol') is not None else None
        
        # If the ticker doesn't match, skip this form
        if issuer_ticker and issuer_ticker.upper() != ticker.upper():
            return []
        
        # Extract information about the reporting owner
        owner_element = root.find(f'.//{ns}reportingOwner')
        owner_name = owner_element.find(f'.//{ns}rptOwnerName').text if owner_element is not None and owner_element.find(f'.//{ns}rptOwnerName') is not None else None
        
        # Check if owner is a director
        is_director = False
        relationship = owner_element.find(f'.//{ns}reportingOwnerRelationship') if owner_element is not None else None
        if relationship is not None and relationship.find(f'.//{ns}isDirector') is not None and relationship.find(f'.//{ns}isDirector').text.upper() == 'TRUE':
            is_director = True
        
        # Get owner's title
        title = None
        if relationship is not None and relationship.find(f'.//{ns}officerTitle') is not None:
            title = relationship.find(f'.//{ns}officerTitle').text
        
        # Extract filing date
        filing_date_elem = root.find(f'.//{ns}periodOfReport')
        filing_date = filing_date_elem.text if filing_date_elem is not None else None
        if filing_date:
            filing_date = datetime.strptime(filing_date, '%Y-%m-%d').strftime('%Y-%m-%d')
        
        # Extract transactions
        transactions = []
        non_derivative_table = root.find(f'.//{ns}nonDerivativeTable')
        
        if non_derivative_table is not None:
            transaction_elements = non_derivative_table.findall(f'.//{ns}nonDerivativeTransaction')
            
            for transaction in transaction_elements:
                # Extract transaction details
                transaction_date_elem = transaction.find(f'.//{ns}transactionDate/{ns}value') if transaction.find(f'.//{ns}transactionDate') is not None else None
                transaction_date = transaction_date_elem.text if transaction_date_elem is not None else None
                
                if transaction_date:
                    transaction_date = datetime.strptime(transaction_date, '%Y-%m-%d').strftime('%Y-%m-%d')
                
                # Security information
                security_title_elem = transaction.find(f'.//{ns}securityTitle/{ns}value') if transaction.find(f'.//{ns}securityTitle') is not None else None
                security_title = security_title_elem.text if security_title_elem is not None else None
                
                # Transaction amounts
                shares_elem = transaction.find(f'.//{ns}transactionAmounts/{ns}transactionShares/{ns}value') if transaction.find(f'.//{ns}transactionAmounts') is not None else None
                price_elem = transaction.find(f'.//{ns}transactionAmounts/{ns}transactionPricePerShare/{ns}value') if transaction.find(f'.//{ns}transactionAmounts') is not None else None
                
                shares = float(shares_elem.text) if shares_elem is not None and shares_elem.text else None
                price = float(price_elem.text) if price_elem is not None and price_elem.text else None
                
                # Transaction code (P=Purchase, S=Sale, etc.)
                transaction_code_elem = transaction.find(f'.//{ns}transactionAmounts/{ns}transactionAcquiredDisposedCode/{ns}value') if transaction.find(f'.//{ns}transactionAmounts') is not None else None
                transaction_code = transaction_code_elem.text if transaction_code_elem is not None else None
                
                # If it's a sale, make shares negative
                if transaction_code == 'D':  # D = Disposed (sold)
                    shares = -shares if shares else None
                
                # Shares owned after transaction
                shares_owned_after_elem = transaction.find(f'.//{ns}postTransactionAmounts/{ns}sharesOwnedFollowingTransaction/{ns}value') if transaction.find(f'.//{ns}postTransactionAmounts') is not None else None
                shares_owned_after = float(shares_owned_after_elem.text) if shares_owned_after_elem is not None and shares_owned_after_elem.text else None
                
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
        logger.error(f"Error parsing Form 4 file {file_path}: {e}", module="sec_edgar_scraper")
        return []