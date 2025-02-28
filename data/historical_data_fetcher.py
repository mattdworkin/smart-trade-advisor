# data/historical_data_fetcher.py
import yfinance as yf
import pandas as pd
import numpy as np
import talib
import argparse
import os
import logging
import time
from typing import List, Dict
from bs4 import BeautifulSoup  # For S&P 500 ticker scraping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import pandas_validator as pv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('historical_data.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class DataValidationSchema:
    required_columns: List[str] = ['Open', 'High', 'Low', 'Close', 'Volume']
    min_rows: int = 100
    max_missing_pct: float = 0.1

class HistoricalDataFetcher:
    def __init__(self, max_workers: int = 4):
        self.data_dir = "data/historical"
        os.makedirs(self.data_dir, exist_ok=True)
        self.max_workers = max_workers
        self.validation_schema = DataValidationSchema()

    def fetch_sp500_tickers(self) -> List[str]:
        """Scrape current S&P 500 constituents from Wikipedia"""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            tables = pd.read_html(url)
            df = tables[0]
            tickers = df['Symbol'].tolist()
            logging.info(f"Fetched {len(tickers)} S&P 500 tickers")
            return [t.replace('.', '-') for t in tickers]  # Yahoo Finance uses dashes
        except Exception as e:
            logging.error(f"Failed to fetch S&P 500 tickers: {e}")
            raise

    def fetch_historical_data(self, ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
        """Fetch historical data with retry logic"""
        max_retries = 3
        backoff_factor = 2
        
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False,
                    threads=True
                )
                if data.empty:
                    raise ValueError(f"No data found for {ticker}")
                
                data = self._add_technical_indicators(data)
                data = self._clean_data(data)
                return data
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to fetch {ticker} after {max_retries} attempts: {e}")
                    return pd.DataFrame()
                sleep_time = backoff_factor ** attempt
                logging.warning(f"Retrying {ticker} in {sleep_time}s... (Attempt {attempt + 1})")
                time.sleep(sleep_time)

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add 25 essential technical indicators"""
        # Price Transformations
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = talib.MACD(data['Close'])
        data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
        
        # Momentum Indicators
        data['Stoch_K'], data['Stoch_D'] = talib.STOCH(data['High'], data['Low'], data['Close'])
        data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
        
        # Volatility Indicators
        data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
        data['NATR'] = talib.NATR(data['High'], data['Low'], data['Close'], timeperiod=14)
        
        # Volume Indicators
        data['OBV'] = talib.OBV(data['Close'], data['Volume'])
        data['AD'] = talib.AD(data['High'], data['Low'], data['Close'], data['Volume'])
        
        # Cycle Indicators
        data['HT_DCPERIOD'] = talib.HT_DCPERIOD(data['Close'])
        
        # Pattern Recognition
        data['CDLDOJI'] = talib.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])
        
        return data

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and anomalies"""
        # Forward fill missing values
        data.ffill(inplace=True)
        
        # Remove any remaining NaN
        data.dropna(inplace=True)
        
        # Remove outliers using Z-score
        cols = data.columns
        for col in cols:
            if data[col].dtype in [np.float64, np.int64]:
                z = np.abs((data[col] - data[col].mean()) / data[col].std())
                data = data[z < 3]
                
        return data

    def save_data(self, ticker: str, data: pd.DataFrame):
        """Save to compressed parquet format"""
        fname = os.path.join(self.data_dir, f"{ticker}.parquet")
        data.to_parquet(fname, compression='brotli')
        logging.info(f"Saved {ticker} data ({len(data)} rows) to {fname}")

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate fetched data meets requirements"""
        try:
            # Check required columns
            if not all(col in data.columns for col in self.validation_schema.required_columns):
                logging.error("Missing required columns")
                return False
                
            # Check minimum rows
            if len(data) < self.validation_schema.min_rows:
                logging.error("Insufficient data rows")
                return False
                
            # Check missing data percentage
            missing_pct = data[self.validation_schema.required_columns].isnull().mean().mean()
            if missing_pct > self.validation_schema.max_missing_pct:
                logging.error(f"Too many missing values: {missing_pct:.2%}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Data validation error: {e}")
            return False

    def process_ticker_batch(self, tickers: List[str], start: str, end: str, interval: str):
        """Process a batch of tickers in parallel"""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.fetch_historical_data, ticker, start, end, interval)
                for ticker in tickers
            ]
            for future in futures:
                try:
                    data = future.result()
                    if not data.empty and self.validate_data(data):
                        results.append(data)
                except Exception as e:
                    logging.error(f"Failed to process ticker batch: {e}")
        return results

def main():
    parser = argparse.ArgumentParser(description='Fetch historical stock data')
    parser.add_argument('--start', type=str, default='2000-01-01',
                       help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, default=pd.Timestamp.today().strftime('%Y-%m-%d'),
                       help='End date in YYYY-MM-DD format')
    parser.add_argument('--interval', type=str, default='1d',
                       choices=['1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo'],
                       help='Data interval')
    parser.add_argument('--max_tickers', type=int, default=None,
                       help='Maximum number of tickers to process (for testing)')
    
    args = parser.parse_args()
    
    fetcher = HistoricalDataFetcher()
    
    try:
        tickers = fetcher.fetch_sp500_tickers()
        if args.max_tickers:
            tickers = tickers[:args.max_tickers]
            
        # Process tickers in batches
        batch_size = 10
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}")
            results = fetcher.process_ticker_batch(batch, args.start, args.end, args.interval)
            
            for data in results:
                if not data.empty:
                    fetcher.save_data(data['Ticker'].iloc[0], data)
            
            time.sleep(1)  # Rate limiting between batches
            
    except Exception as e:
        logging.error(f"Critical error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()