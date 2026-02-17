"""
Data Loader Module
------------------
Responsible for fetching external financial data.
1. Scrapes the current S&P 500 list from Wikipedia.
2. Downloads historical price data using Yahoo Finance (yfinance).
"""

import requests
import pandas as pd
import yfinance as yf
from typing import List
from io import StringIO  # Required for newer Pandas versions

def get_sp500_tickers() -> List[str]:
    """
    Scrapes Wikipedia to get the current list of S&P 500 tickers.
    
    Returns:
        List[str]: A list of ticker symbols (e.g., ['AAPL', 'MSFT', ...]).
        
    Raises:
        RuntimeError: If scraping fails or the table cannot be found.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # User-Agent header is required to avoid 403 Forbidden error from Wikipedia :-)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # FIX 1: Use StringIO (Pandas 2.0+ deprecation fix)
        # FIX 2: Target specific table ID "constituents" for robustness
        tables = pd.read_html(StringIO(response.text), attrs={"id": "constituents"})
        
        if not tables:
            raise ValueError("Could not find the S&P 500 table on Wikipedia.")
            
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        
        # Clean tickers: Replace dots with hyphens for yfinance compatibility (e.g., BRK.B -> BRK-B)
        return [ticker.replace('.', '-') for ticker in tickers]
    
    except Exception as e:
        raise RuntimeError(f"Error fetching S&P 500 tickers: {e}")


def get_historical_data(
    tickers: List[str], 
    period: str = '2y',
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Downloads historical Adjusted Close prices for a list of tickers.
    Uses bulk download for performance optimization.

    Args:
        tickers (List[str]): List of symbol strings.
        period (str): Data period to download (e.g., '1y', '2y', 'max').
        interval (str): Data interval (e.g., '1d', '1wk').

    Returns:
        pd.DataFrame: DataFrame where columns are Tickers and index is Date.
    """
    if not tickers:
        raise ValueError("Ticker list cannot be empty.")
    
    print(f"⏳ Downloading data for {len(tickers)} stocks...")
    
    try:
        # Bulk download using yfinance
        # auto_adjust=True -> 'Close' column will contain split/dividend adjusted price
        # group_by='column' -> Easier to extract the specific column for all stocks
        # threads=True -> Enables parallel downloading
        data = yf.download(
            tickers, 
            period=period, 
            interval=interval, 
            group_by='column', 
            auto_adjust=True, 
            threads=True,
            progress=False
        )
        
        # CASE A: Single Ticker (yfinance returns a simple DataFrame, not MultiIndex)
        if len(tickers) == 1:
            ticker = tickers[0]
            if 'Close' in data.columns:
                df = data[['Close']].copy()
                df.columns = [ticker]
                return df
            else:
                raise ValueError(f"Missing 'Close' column for ticker {ticker}")

        # CASE B: Multiple Tickers (yfinance returns MultiIndex DataFrame)
        # Structure: Level 0 = Price Type (Close, Open...), Level 1 = Ticker
        
        if 'Close' in data.columns:
            # Extract only the Close prices (which are adjusted due to auto_adjust=True)
            df_close = data['Close']
            
            # Drop columns where all values are NaN (failed downloads)
            df_close = df_close.dropna(axis=1, how='all')
            
            if df_close.empty:
                raise ValueError("Failed to download any data (all values are NaN).")
                
            print(f"✅ Successfully downloaded data for {len(df_close.columns)} stocks.")
            return df_close
        else:
            raise KeyError("Downloaded data does not contain 'Close' column.")

    except Exception as e:
        print(f"❌ Error in get_historical_data: {e}")
        return pd.DataFrame()