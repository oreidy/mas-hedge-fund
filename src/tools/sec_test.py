from secedgar import filings, FilingType
from datetime import date, timedelta, datetime
import os
import glob
import sys
import json
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from typing import List, Optional


# Define a simple InsiderTrade class right in this file
@dataclass
class InsiderTrade:
    ticker: str
    issuer: Optional[str] = None
    name: Optional[str] = None
    title: Optional[str] = None
    is_board_director: bool = False
    transaction_date: Optional[str] = None
    transaction_shares: Optional[float] = None
    transaction_price_per_share: Optional[float] = None
    transaction_value: Optional[float] = None
    shares_owned_before_transaction: Optional[float] = None
    shares_owned_after_transaction: Optional[float] = None
    security_title: Optional[str] = None
    filing_date: Optional[str] = None
    
    def model_dump(self):
        return self.__dict__
    
# Define the parsing function right in this file
def parse_form4_file(file_path: str, ticker: str) -> List[InsiderTrade]:
    """
    Parse a Form 4 txt file to extract insider trading information.
    
    Args:
        file_path: Path to the Form 4 txt file
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
            print(f"No XML tags found in {file_path}")
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
        print(f"Error parsing Form 4 file {file_path}: {e}")
        return []
    


    
# Add the parent directory to the Python path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the path to Downloads folder
downloads_path = os.path.expanduser("~/Downloads/sec_edgar_filings")

# Create the directory if it doesn't exist
os.makedirs(downloads_path, exist_ok=True)

end_date = date(2024, 5, 21)
start_date = end_date - timedelta(days=2)

# Download the latest 10-Q filing for Apple (ticker: AAPL)
my_filings = filings(
    cik_lookup="JPM, AAPL",
    filing_type=FilingType.FILING_4,
    user_agent="Oliver Reidy (oliver.reidy@uzh.ch)",
    start_date=start_date,
    end_date=end_date
)

# Save to the specified directory
print(f"Saving filings to: {downloads_path}")
my_filings.save(downloads_path)
print("Filings saved successfully!")

# Print what was saved (optional)
print("\nDirectory contents:")
for root, dirs, files in os.walk(downloads_path):
    print(f"Directory: {root}")
    print(f"Subdirectories: {dirs}")
    print(f"Number of files: {len(files)}")

    # Now parse the txt files
filings_path = os.path.join(downloads_path, "JPM", "4")
if os.path.exists(filings_path):
    print(f"\nParsing TXT files in: {filings_path}")
    
    # Find all txt files
    txt_files = glob.glob(os.path.join(filings_path, "*.txt"))
    print(f"Found {len(txt_files)} TXT files")
    
    # Parse each file
    total_trades = 0
    all_trades = []
    
    for txt_file in txt_files:
        print(f"\nProcessing file: {os.path.basename(txt_file)}")
        
        # Call the parse_form4_file function
        trades = parse_form4_file(txt_file, "JPM")
        
        # Print results
        if trades:
            total_trades += len(trades)
            all_trades.extend(trades)
            print(f"Successfully parsed {len(trades)} trades:")
            for i, trade in enumerate(trades):
                print(f"Trade {i+1}:")
                print(f"  Insider: {trade.name} ({trade.title or 'N/A'})")
                print(f"  Date: {trade.transaction_date}")
                shares_display = f"{trade.transaction_shares:,.0f}" if trade.transaction_shares else "N/A"
                price_display = f"${trade.transaction_price_per_share:,.2f}" if trade.transaction_price_per_share else "N/A"
                value_display = f"${trade.transaction_value:,.2f}" if trade.transaction_value else "N/A"
                print(f"  Shares: {shares_display}")
                print(f"  Price: {price_display}")
                print(f"  Value: {value_display}")
        else:
            print("No trades found in this file")
    
    print(f"\nTotal trades found across all files: {total_trades}")
    
    # Optionally, you can save all the parsed trades to a JSON file
    if all_trades:
        import json
        
        cache_file = os.path.join(downloads_path, f"{"JPM"}_insider_trades.json")
        with open(cache_file, 'w') as f:
            json.dump([trade.model_dump() for trade in all_trades], f)
        print(f"Saved all trades to: {cache_file}")
else:
    print(f"Error: Directory {filings_path} does not exist")