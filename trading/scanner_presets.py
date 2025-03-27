import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ScannerPresets:
    """
    Provides specialized market scanners for various trading setups
    """
    
    def __init__(self, data_fetcher=None, realtime_stream=None):
        """
        Initialize scanner with data sources
        
        Args:
            data_fetcher: Historical data fetcher instance
            realtime_stream: Realtime data stream instance
        """
        self.data_fetcher = data_fetcher
        self.realtime_stream = realtime_stream
        self.scan_results_dir = 'results/scans'
        os.makedirs(self.scan_results_dir, exist_ok=True)
    
    def scan_gap_plays(self, min_gap_percent=3.0, max_price=500.0, min_volume=500000) -> List[Dict[str, Any]]:
        """
        Scan for stocks with significant pre-market or overnight gaps
        
        Args:
            min_gap_percent: Minimum gap percentage to include
            max_price: Maximum stock price to include
            min_volume: Minimum average volume to include
            
        Returns:
            List of gap play candidates
        """
        logger.info(f"Scanning for gap plays (min gap: {min_gap_percent}%)")
        
        try:
            import yfinance as yf
            
            # Get list of liquid stocks to scan
            symbols = self._get_scan_universe(max_symbols=300)
            
            gap_plays = []
            
            # Process in batches
            batch_size = 25
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                
                # Get data for each symbol
                try:
                    # Download batch data - 5 days of daily data to calculate gaps
                    data = yf.download(
                        " ".join(batch),
                        period="5d",
                        group_by='ticker',
                        prepost=True,  # Include pre/post market data if available
                        progress=False
                    )
                    
                    for symbol in batch:
                        try:
                            # Get symbol data
                            if isinstance(data, pd.DataFrame) and ('Adj Close' in data.columns or (symbol in data and 'Adj Close' in data[symbol].columns)):
                                # Single symbol case
                                if 'Adj Close' in data.columns:
                                    symbol_data = data
                                # Multi-symbol case
                                else:
                                    symbol_data = data[symbol]
                                
                                if len(symbol_data) >= 2:
                                    # Calculate gap percentage
                                    prev_close = symbol_data['Close'].iloc[-2]
                                    current_open = symbol_data['Open'].iloc[-1]
                                    gap_percent = ((current_open - prev_close) / prev_close) * 100
                                    
                                    # Check if meets criteria
                                    current_price = symbol_data['Close'].iloc[-1]
                                    avg_volume = symbol_data['Volume'].mean()
                                    
                                    if (abs(gap_percent) >= min_gap_percent and 
                                            current_price <= max_price and 
                                            avg_volume >= min_volume):
                                        
                                        gap_plays.append({
                                            'symbol': symbol,
                                            'gap_percent': gap_percent,
                                            'price': current_price,
                                            'volume': avg_volume,
                                            'gap_direction': 'up' if gap_percent > 0 else 'down',
                                            'scan_type': 'gap_play'
                                        })
                                        
                                        logger.info(f"Found gap play: {symbol} with {gap_percent:.2f}% gap")
                                
                        except Exception as e:
                            logger.debug(f"Error processing {symbol} for gap plays: {e}")
                            continue
                
                except Exception as e:
                    logger.warning(f"Error fetching batch data for gap plays: {e}")
            
            # Sort by gap percent (absolute value, descending)
            gap_plays.sort(key=lambda x: abs(x['gap_percent']), reverse=True)
            
            # Save results for later reference
            if gap_plays:
                df = pd.DataFrame(gap_plays)
                filename = os.path.join(self.scan_results_dir, f"gap_plays_{datetime.now().strftime('%Y%m%d')}.csv")
                df.to_csv(filename, index=False)
                logger.info(f"Saved {len(gap_plays)} gap plays to {filename}")
            
            return gap_plays
            
        except Exception as e:
            logger.error(f"Error scanning for gap plays: {e}")
            return []
    
    def scan_breakouts(self, lookback_days=20, volume_surge=1.5, min_price=5.0) -> List[Dict[str, Any]]:
        """
        Scan for stocks breaking out of consolidation patterns
        
        Args:
            lookback_days: Days to look back for consolidation pattern
            volume_surge: Minimum ratio of current volume to average
            min_price: Minimum stock price to include
            
        Returns:
            List of breakout candidates
        """
        logger.info(f"Scanning for breakouts (lookback: {lookback_days} days)")
        
        try:
            import yfinance as yf
            
            # Get list of liquid stocks to scan
            symbols = self._get_scan_universe(max_symbols=300)
            
            breakouts = []
            
            # Process in batches
            batch_size = 25
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                
                # Get data for each symbol
                try:
                    # Download batch data - lookback period plus some extra days
                    data = yf.download(
                        " ".join(batch),
                        period=f"{lookback_days + 10}d",
                        group_by='ticker',
                        progress=False
                    )
                    
                    for symbol in batch:
                        try:
                            # Get symbol data
                            if isinstance(data, pd.DataFrame) and ('Adj Close' in data.columns or (symbol in data and 'Adj Close' in data[symbol].columns)):
                                # Single symbol case
                                if 'Adj Close' in data.columns:
                                    symbol_data = data
                                # Multi-symbol case
                                else:
                                    symbol_data = data[symbol]
                                
                                if len(symbol_data) >= lookback_days:
                                    # Calculate breakout metrics
                                    
                                    # Get recent data for analysis
                                    recent_data = symbol_data.iloc[-lookback_days:]
                                    
                                    # 1. Check for a consolidation pattern (tight range)
                                    # Calculate the range as % of price for each day
                                    recent_data['day_range_pct'] = (recent_data['High'] - recent_data['Low']) / recent_data['Low'] * 100
                                    
                                    # Calculate the average range percentage
                                    avg_range_pct = recent_data['day_range_pct'].mean()
                                    
                                    # 2. Check for a breakout - price breaking above recent resistance
                                    resistance_level = recent_data['High'].iloc[:-1].max()  # Max high excluding today
                                    current_price = recent_data['Close'].iloc[-1]
                                    
                                    # 3. Check for volume surge
                                    avg_volume = recent_data['Volume'].iloc[:-1].mean()  # Average volume excluding today
                                    current_volume = recent_data['Volume'].iloc[-1]
                                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                                    
                                    # Criteria for a breakout:
                                    # - Current price > resistance level
                                    # - Volume ratio > specified threshold
                                    # - Price > minimum threshold
                                    if (current_price > resistance_level * 1.02 and  # 2% above resistance
                                            volume_ratio >= volume_surge and 
                                            current_price >= min_price):
                                        
                                        # Calculate additional breakout metrics
                                        breakout_percent = ((current_price - resistance_level) / resistance_level) * 100
                                        
                                        breakouts.append({
                                            'symbol': symbol,
                                            'breakout_percent': breakout_percent,
                                            'price': current_price,
                                            'resistance_level': resistance_level,
                                            'volume_ratio': volume_ratio,
                                            'avg_range_pct': avg_range_pct,
                                            'scan_type': 'breakout'
                                        })
                                        
                                        logger.info(f"Found breakout: {symbol} breaking {breakout_percent:.2f}% above resistance")
                                
                        except Exception as e:
                            logger.debug(f"Error processing {symbol} for breakouts: {e}")
                            continue
                
                except Exception as e:
                    logger.warning(f"Error fetching batch data for breakouts: {e}")
            
            # Sort by breakout percent (descending)
            breakouts.sort(key=lambda x: x['breakout_percent'], reverse=True)
            
            # Save results for later reference
            if breakouts:
                df = pd.DataFrame(breakouts)
                filename = os.path.join(self.scan_results_dir, f"breakouts_{datetime.now().strftime('%Y%m%d')}.csv")
                df.to_csv(filename, index=False)
                logger.info(f"Saved {len(breakouts)} breakouts to {filename}")
            
            return breakouts
            
        except Exception as e:
            logger.error(f"Error scanning for breakouts: {e}")
            return []
    
    def scan_oversold_reversals(self, max_rsi=30, max_price=200.0) -> List[Dict[str, Any]]:
        """
        Scan for oversold stocks showing signs of reversal
        
        Args:
            max_rsi: Maximum RSI value to consider as oversold
            max_price: Maximum stock price to include
            
        Returns:
            List of oversold reversal candidates
        """
        logger.info(f"Scanning for oversold reversals (max RSI: {max_rsi})")
        
        try:
            import yfinance as yf
            
            # Get list of stocks to scan
            symbols = self._get_scan_universe(max_symbols=300)
            
            reversals = []
            
            # Process in batches
            batch_size = 25
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                
                # Get data for each symbol
                try:
                    # Download batch data
                    data = yf.download(
                        " ".join(batch),
                        period="30d",  # Get enough data for RSI calculation
                        group_by='ticker',
                        progress=False
                    )
                    
                    for symbol in batch:
                        try:
                            # Get symbol data
                            if isinstance(data, pd.DataFrame) and ('Adj Close' in data.columns or (symbol in data and 'Adj Close' in data[symbol].columns)):
                                # Single symbol case
                                if 'Adj Close' in data.columns:
                                    symbol_data = data
                                # Multi-symbol case
                                else:
                                    symbol_data = data[symbol]
                                
                                if len(symbol_data) >= 14:  # Need at least 14 days for RSI
                                    # Calculate RSI
                                    from models.model_trainer import ModelTrainer
                                    model_trainer = ModelTrainer()
                                    
                                    feature_data = model_trainer._engineer_features(
                                        symbol_data.rename(columns={
                                            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                                        }),
                                        ['rsi', 'bollinger'],
                                        14
                                    )
                                    
                                    if 'rsi_14' in feature_data.columns:
                                        # Get the current RSI
                                        current_rsi = feature_data['rsi_14'].iloc[-1]
                                        
                                        # Check for reversal pattern
                                        # 1. Current RSI is oversold
                                        # 2. Price is showing signs of reversal (higher low or bullish candlestick)
                                        
                                        current_price = symbol_data['Close'].iloc[-1]
                                        prev_price = symbol_data['Close'].iloc[-2]
                                        
                                        # Simple reversal check: Current price > previous price
                                        price_reversal = current_price > prev_price
                                        
                                        # Check if price is near Bollinger Band lower band
                                        near_lower_band = False
                                        if 'bollinger_lower' in feature_data.columns:
                                            lower_band = feature_data['bollinger_lower'].iloc[-1]
                                            near_lower_band = (current_price / lower_band - 1) < 0.03  # Within 3% of lower band
                                        
                                        if (current_rsi <= max_rsi and 
                                                price_reversal and 
                                                current_price <= max_price and
                                                (near_lower_band or current_rsi < 25)):  # Stronger criteria if not near band
                                            
                                            reversals.append({
                                                'symbol': symbol,
                                                'rsi': current_rsi,
                                                'price': current_price,
                                                'price_change_pct': ((current_price - prev_price) / prev_price) * 100,
                                                'near_support': near_lower_band,
                                                'scan_type': 'oversold_reversal'
                                            })
                                            
                                            logger.info(f"Found oversold reversal: {symbol} with RSI {current_rsi:.2f}")
                                
                        except Exception as e:
                            logger.debug(f"Error processing {symbol} for oversold reversals: {e}")
                            continue
                
                except Exception as e:
                    logger.warning(f"Error fetching batch data for oversold reversals: {e}")
            
            # Sort by RSI (ascending - most oversold first)
            reversals.sort(key=lambda x: x['rsi'])
            
            # Save results for later reference
            if reversals:
                df = pd.DataFrame(reversals)
                filename = os.path.join(self.scan_results_dir, f"oversold_reversals_{datetime.now().strftime('%Y%m%d')}.csv")
                df.to_csv(filename, index=False)
                logger.info(f"Saved {len(reversals)} oversold reversals to {filename}")
            
            return reversals
            
        except Exception as e:
            logger.error(f"Error scanning for oversold reversals: {e}")
            return []
    
    def scan_earnings_movers(self, min_move_percent=5.0, days_from_earnings=3) -> List[Dict[str, Any]]:
        """
        Scan for stocks with significant moves after recent earnings
        
        Args:
            min_move_percent: Minimum price move percentage
            days_from_earnings: Maximum days since earnings
            
        Returns:
            List of earnings movers
        """
        logger.info(f"Scanning for earnings movers (min move: {min_move_percent}%)")
        
        try:
            import yfinance as yf
            
            # Get list of stocks to scan
            symbols = self._get_scan_universe(max_symbols=200)
            
            earnings_movers = []
            
            # Process in batches
            batch_size = 25
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                
                for symbol in batch:
                    try:
                        # Get ticker info
                        ticker = yf.Ticker(symbol)
                        
                        # Get earnings date
                        try:
                            # Get calendar data
                            calendar = ticker.calendar
                            
                            if calendar is not None and not calendar.empty:
                                earnings_date = calendar.iloc[0, 0]  # First row, first column is earnings date
                                
                                # Convert to datetime if needed
                                if not isinstance(earnings_date, (datetime, pd.Timestamp)):
                                    continue
                                
                                # Check if earnings are recent
                                days_since_earnings = (datetime.now() - pd.to_datetime(earnings_date)).days
                                
                                if days_since_earnings <= days_from_earnings:
                                    # Get price data around earnings
                                    hist = ticker.history(period="10d")
                                    
                                    if len(hist) >= 2:
                                        # Find the price before and after earnings
                                        earnings_date_str = pd.to_datetime(earnings_date).strftime('%Y-%m-%d')
                                        
                                        # Find closest date before and after earnings
                                        hist_dates = hist.index.strftime('%Y-%m-%d').to_list()
                                        
                                        # Find date index closest to earnings date
                                        try:
                                            earnings_idx = hist_dates.index(earnings_date_str)
                                        except ValueError:
                                            # Find the closest date after earnings
                                            for idx, date in enumerate(hist_dates):
                                                if date >= earnings_date_str:
                                                    earnings_idx = idx
                                                    break
                                            else:
                                                earnings_idx = 0
                                        
                                        # Get price before earnings (day before or closest)
                                        before_idx = max(0, earnings_idx - 1)
                                        before_price = hist['Close'].iloc[before_idx]
                                        
                                        # Get price after earnings (most recent)
                                        after_price = hist['Close'].iloc[-1]
                                        
                                        # Calculate move percentage
                                        move_percent = ((after_price - before_price) / before_price) * 100
                                        
                                        if abs(move_percent) >= min_move_percent:
                                            earnings_movers.append({
                                                'symbol': symbol,
                                                'move_percent': move_percent,
                                                'price': after_price,
                                                'earnings_date': earnings_date_str,
                                                'days_since_earnings': days_since_earnings,
                                                'move_direction': 'up' if move_percent > 0 else 'down',
                                                'scan_type': 'earnings_mover'
                                            })
                                            
                                            logger.info(f"Found earnings mover: {symbol} with {move_percent:.2f}% move after earnings")
                        
                        except Exception as e:
                            logger.debug(f"Error processing earnings data for {symbol}: {e}")
                            continue
                    
                    except Exception as e:
                        logger.debug(f"Error processing {symbol} for earnings movers: {e}")
                        continue
            
            # Sort by move percent (absolute value, descending)
            earnings_movers.sort(key=lambda x: abs(x['move_percent']), reverse=True)
            
            # Save results for later reference
            if earnings_movers:
                df = pd.DataFrame(earnings_movers)
                filename = os.path.join(self.scan_results_dir, f"earnings_movers_{datetime.now().strftime('%Y%m%d')}.csv")
                df.to_csv(filename, index=False)
                logger.info(f"Saved {len(earnings_movers)} earnings movers to {filename}")
            
            return earnings_movers
            
        except Exception as e:
            logger.error(f"Error scanning for earnings movers: {e}")
            return []
    
    def _get_scan_universe(self, max_symbols=300) -> List[str]:
        """Get the universe of symbols to scan"""
        try:
            # Try using S&P 500 + NASDAQ 100 + Dow 30
            import yfinance as yf
            
            symbols = []
            
            # Try to get S&P 500 components
            try:
                sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
                sp500_symbols = sp500['Symbol'].tolist()
                symbols.extend(sp500_symbols)
            except:
                logger.warning("Could not fetch S&P 500 symbols")
            
            # Add some liquid ETFs
            etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY', 'XLB', 'ARKK', 'ARKG']
            symbols.extend(etfs)
            
            # Make symbols unique and limit size
            symbols = list(set(symbols))[:max_symbols]
            
            # Clean symbols (replace dots with dashes for Yahoo Finance)
            symbols = [s.replace('.', '-') for s in symbols]
            
            return symbols
        
        except Exception as e:
            logger.warning(f"Error getting scan universe: {e}")
            
            # Fallback to a default list of liquid stocks
            return [
                # Tech
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "TSLA", "INTC", "CRM", 
                # Finance
                "JPM", "BAC", "GS", "MS", "V", "MA", "PYPL", "BRK-B", "C", "WFC",
                # Healthcare
                "JNJ", "PFE", "MRK", "ABBV", "UNH", "CVS", "MDT", "LLY", "TMO", "ABT",
                # Consumer
                "WMT", "PG", "KO", "PEP", "MCD", "SBUX", "NKE", "DIS", "COST", "HD",
                # Energy
                "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "OXY", "BP", "MPC",
                # ETFs
                "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP"
            ] 