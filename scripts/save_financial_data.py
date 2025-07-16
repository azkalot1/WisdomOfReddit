import requests
import yfinance as yf
import pandas as pd
import time
import os
import argparse
from datetime import datetime, date
from typing import List, Optional
import numpy as np

def get_all_tickers_fmp(api_key: str) -> List[str]:
    """
    Get all available tickers using Financial Modeling Prep API
    Requires free API key from https://financialmodelingprep.com/
    """
    
    # Get all available stocks
    url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        stocks_data = response.json()
        
        # Filter for US stocks only (optional)
        us_tickers = []
        for stock in stocks_data:
            symbol = stock.get('symbol', '')
            exchange = stock.get('exchangeShortName', '')
            
            # Filter for major US exchanges and reasonable symbol length
            if (exchange in ['NASDAQ', 'NYSE', 'AMEX'] and 
                len(symbol) <= 5 and 
                symbol.isalpha()):  # Only alphabetic symbols
                us_tickers.append(symbol)
        
        print(f"Found {len(us_tickers)} US tickers from FMP")
        return us_tickers
        
    except Exception as e:
        print(f"Error fetching from FMP: {e}")
        return []

def validate_date(date_string: str) -> str:
    """Validate date format YYYY-MM-DD"""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return date_string
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_string}. Use YYYY-MM-DD")

def validate_file_path(file_path: str) -> str:
    """Validate and create directory if needed"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Cannot create directory {directory}: {e}")
    
    # Check if we can write to the location
    try:
        test_file = file_path + ".test"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Cannot write to {file_path}: {e}")
    
    return file_path

def download_financial_data_batch(
    tickers: List[str],
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
    batch_size: int = 100,
    delay_between_batches: float = 1.0,
    auto_adjust: bool = True,
    save_progress: bool = True,
    progress_file: str = "financial_data_progress.parquet"
) -> pd.DataFrame:
    """
    Download financial data for a list of tickers using yfinance with batching
    
    Args:
        tickers: List of ticker symbols to download
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)
        batch_size: Number of tickers to process per batch
        delay_between_batches: Seconds to wait between batches
        auto_adjust: Whether to use auto-adjusted prices
        save_progress: Whether to save progress incrementally
        progress_file: File to save progress to
        
    Returns:
        pd.DataFrame: Combined financial data for all tickers in long format
    """
    
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
    print(f"Batch size: {batch_size}, Auto-adjust: {auto_adjust}")
    
    all_data_frames = []
    failed_tickers = []
    successful_tickers = []
    
    # Calculate total batches
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"\nProcessing batch {batch_num}/{total_batches}: {len(batch_tickers)} tickers")
        print(f"Tickers: {batch_tickers[:5]}{'...' if len(batch_tickers) > 5 else ''}")
        
        try:
            # Download data for this batch
            batch_data_raw = yf.download(
                batch_tickers,
                start=start_date,
                end=end_date,
                auto_adjust=auto_adjust,
                progress=False,
                threads=True,
                group_by='ticker'
            )
            
            if batch_data_raw.empty:
                print(f"  ✗ No data returned for batch")
                failed_tickers.extend(batch_tickers)
                continue
            
            # Process each ticker individually to create long format
            batch_processed_list = []
            
            for ticker in batch_tickers:
                try:
                    if len(batch_tickers) == 1:
                        # Single ticker case
                        ticker_data = batch_data_raw.copy()
                    else:
                        # Multi-ticker case - extract specific ticker
                        if ticker in batch_data_raw.columns.get_level_values(0):
                            ticker_data = batch_data_raw[ticker].copy()
                        else:
                            print(f"    ⚠ No data for {ticker}")
                            failed_tickers.append(ticker)
                            continue
                    
                    # Check if ticker has any valid data
                    if ticker_data.empty or ticker_data.isna().all().all():
                        print(f"    ⚠ Empty/invalid data for {ticker}")
                        failed_tickers.append(ticker)
                        continue
                    
                    # Reset index to get Date as a column
                    ticker_data = ticker_data.reset_index()
                    
                    # Add ticker column
                    ticker_data['Ticker'] = ticker
                    
                    # Rename Date column for consistency
                    if 'Date' in ticker_data.columns:
                        ticker_data.rename(columns={'Date': 'date'}, inplace=True)
                    
                    # Ensure date is datetime
                    ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                    
                    # Remove rows where all price columns are NaN
                    price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if auto_adjust:
                        price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    else:
                        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                    
                    # Only keep columns that exist
                    existing_price_cols = [col for col in price_columns if col in ticker_data.columns]
                    
                    if existing_price_cols:
                        # Remove rows where all price data is NaN
                        ticker_data = ticker_data.dropna(subset=existing_price_cols, how='all')
                    
                    if not ticker_data.empty:
                        batch_processed_list.append(ticker_data)
                        successful_tickers.append(ticker)
                        print(f"    ✓ {ticker}: {len(ticker_data)} rows")
                    else:
                        print(f"    ⚠ No valid data after cleaning for {ticker}")
                        failed_tickers.append(ticker)
                        
                except Exception as e:
                    print(f"    ✗ Error processing {ticker}: {str(e)}")
                    failed_tickers.append(ticker)
            
            # Combine processed tickers from this batch
            if batch_processed_list:
                batch_combined = pd.concat(batch_processed_list, ignore_index=True)
                all_data_frames.append(batch_combined)
                print(f"  ✓ Batch {batch_num} processed: {len(batch_processed_list)} successful tickers")
            else:
                print(f"  ✗ No successful tickers in batch {batch_num}")
                
        except Exception as e:
            print(f"  ✗ Error processing batch {batch_num}: {str(e)}")
            failed_tickers.extend(batch_tickers)
        
        # Save progress periodically
        if save_progress and batch_num % 5 == 0 and all_data_frames:
            temp_combined = pd.concat(all_data_frames, ignore_index=True)
            progress_path = f"{progress_file}_batch_{batch_num}.parquet"
            temp_combined.to_parquet(progress_path, index=False)
            print(f"  💾 Progress saved to {progress_path}")
        
        # Rate limiting - be respectful to Yahoo Finance
        if i + batch_size < len(tickers):
            time.sleep(delay_between_batches)
    
    # Combine all successful downloads
    if all_data_frames:
        print(f"\n🔄 Combining {len(all_data_frames)} batches...")
        final_data = pd.concat(all_data_frames, ignore_index=True)
        
        # Sort by ticker and date for consistency
        final_data = final_data.sort_values(['Ticker', 'date']).reset_index(drop=True)
        
        # Final cleanup - remove any remaining rows with all NaN price data
        price_columns = [col for col in final_data.columns 
                        if col not in ['date', 'Ticker'] and col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        
        if price_columns:
            initial_rows = len(final_data)
            final_data = final_data.dropna(subset=price_columns, how='all')
            if len(final_data) < initial_rows:
                print(f"  🧹 Removed {initial_rows - len(final_data)} rows with no price data")
        
        # Summary statistics
        unique_tickers = final_data['Ticker'].nunique()
        date_range = f"{final_data['date'].min().date()} to {final_data['date'].max().date()}"
        
        print(f"\n✅ Download Complete!")
        print(f"  📈 Total rows: {len(final_data):,}")
        print(f"  🎯 Unique tickers: {unique_tickers:,}")
        print(f"  📅 Date range: {date_range}")
        print(f"  ✅ Successful: {len(set(successful_tickers))}")
        print(f"  ❌ Failed: {len(set(failed_tickers))}")
        
        if failed_tickers:
            unique_failed = list(set(failed_tickers))
            print(f"  Failed tickers sample: {unique_failed[:10]}")
        
        # Display column info
        print(f"  📊 Columns: {list(final_data.columns)}")
        
        return final_data
    
    else:
        print("\n❌ No data was successfully downloaded")
        return pd.DataFrame()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Download financial data for multiple tickers using yfinance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data for specific tickers
  python script.py --tickers AAPL GOOGL MSFT --start-date 2023-01-01 --end-date 2024-01-01 --output data.parquet
  
  # Download from file with ticker list
  python script.py --ticker-file tickers.txt --start-date 2022-01-01 --output financial_data.parquet
  
  # Use FMP API to get all tickers
  python script.py --fmp-api-key YOUR_KEY --start-date 2023-01-01 --max-tickers 500 --output all_stocks.parquet
  
  # Save as CSV instead of parquet
  python script.py --tickers AAPL TSLA --start-date 2023-01-01 --output data.csv --format csv
        """
    )
    
    # Ticker sources (mutually exclusive)
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument(
        '--tickers', 
        nargs='+', 
        help='List of ticker symbols (e.g., AAPL GOOGL MSFT)'
    )
    ticker_group.add_argument(
        '--ticker-file', 
        type=str,
        help='Path to file containing ticker symbols (one per line)'
    )
    ticker_group.add_argument(
        '--fmp-api-key', 
        type=str,
        help='Financial Modeling Prep API key to fetch all available tickers'
    )
    
    # Date range
    parser.add_argument(
        '--start-date', 
        type=validate_date,
        default='2020-01-01',
        help='Start date for data download (YYYY-MM-DD, default: 2020-01-01)'
    )
    parser.add_argument(
        '--end-date', 
        type=validate_date,
        default=date.today().strftime('%Y-%m-%d'),
        help='End date for data download (YYYY-MM-DD, default: today)'
    )
    
    # Output settings
    parser.add_argument(
        '--output', 
        type=validate_file_path,
        default='financial_data.parquet',
        help='Output file path (default: financial_data.parquet)'
    )
    parser.add_argument(
        '--format', 
        choices=['parquet', 'csv'],
        default='parquet',
        help='Output file format (default: parquet)'
    )
    
    # Processing settings
    parser.add_argument(
        '--batch-size', 
        type=int,
        default=50,
        help='Number of tickers to process per batch (default: 50)'
    )
    parser.add_argument(
        '--delay', 
        type=float,
        default=1.0,
        help='Delay between batches in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--max-tickers', 
        type=int,
        help='Maximum number of tickers to process (useful for testing)'
    )
    parser.add_argument(
        '--no-auto-adjust', 
        action='store_true',
        help='Disable auto-adjustment of prices'
    )
    parser.add_argument(
        '--no-progress-save', 
        action='store_true',
        help='Disable saving progress files'
    )
    
    return parser.parse_args()

def load_tickers_from_file(file_path: str) -> List[str]:
    """Load ticker symbols from a text file"""
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        print(f"Loaded {len(tickers)} tickers from {file_path}")
        return tickers
    except Exception as e:
        raise ValueError(f"Error reading ticker file {file_path}: {e}")

def main():
    """Main function with argument parsing"""
    args = parse_arguments()
    
    print("🚀 Financial Data Downloader")
    print("=" * 50)
    
    # Get ticker list based on source
    if args.tickers:
        tickers = [ticker.upper() for ticker in args.tickers]
        print(f"📋 Using provided tickers: {tickers}")
        
    elif args.ticker_file:
        tickers = load_tickers_from_file(args.ticker_file)
        
    elif args.fmp_api_key:
        print("🔍 Fetching tickers from Financial Modeling Prep...")
        tickers = get_all_tickers_fmp(args.fmp_api_key)
        if not tickers:
            print("❌ Failed to fetch tickers from FMP")
            return
    
    # Limit tickers if specified
    if args.max_tickers and len(tickers) > args.max_tickers:
        print(f"📊 Limiting to first {args.max_tickers} tickers")
        tickers = tickers[:args.max_tickers]
    
    # Validate date range
    if args.start_date >= args.end_date:
        print("❌ Error: Start date must be before end date")
        return
    
    print(f"📅 Date range: {args.start_date} to {args.end_date}")
    print(f"💾 Output: {args.output} ({args.format} format)")
    print(f"⚙️  Batch size: {args.batch_size}, Delay: {args.delay}s")
    
    # Create progress file path
    output_dir = os.path.dirname(args.output) or '.'
    progress_file = os.path.join(output_dir, "progress_" + os.path.basename(args.output))
    
    # Download the data
    financial_data = download_financial_data_batch(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        delay_between_batches=args.delay,
        auto_adjust=not args.no_auto_adjust,
        save_progress=not args.no_progress_save,
        progress_file=progress_file
    )
    
    if not financial_data.empty:
        # Save final data
        try:
            if args.format == 'csv':
                financial_data.to_csv(args.output, index=False)
            else:
                financial_data.to_parquet(args.output, index=False)
            
            print(f"\n💾 Final data saved to: {args.output}")
            print(f"📊 File size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")
            
            # Show sample of data
            print(f"\n📋 Sample data:")
            print(financial_data.head(10))
            
        except Exception as e:
            print(f"❌ Error saving file: {e}")
    else:
        print("❌ No data to save")

if __name__ == "__main__":
    main()