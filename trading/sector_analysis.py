import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class SectorAnalysis:
    """
    Analyzes sector performance and rotation for market regime analysis
    """
    
    def __init__(self, data_fetcher=None):
        """
        Initialize sector analysis
        
        Args:
            data_fetcher: Historical data fetcher instance
        """
        self.data_fetcher = data_fetcher
        self.results_dir = 'results/sectors'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define sector ETFs
        self.sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financial',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrial',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLB': 'Materials',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate',
            'XLC': 'Communication Services'
        }
        
        # Define market ETFs for relative performance
        self.market_etf = 'SPY'  # S&P 500 ETF
    
    def analyze_sector_performance(self, lookback_days=60) -> Dict[str, Any]:
        """
        Analyze sector performance and rotation
        
        Args:
            lookback_days: Number of days to analyze
            
        Returns:
            Dictionary with sector performance data
        """
        logger.info(f"Analyzing sector performance over {lookback_days} days")
        
        try:
            import yfinance as yf
            
            # Get all sectors plus market ETF
            symbols = list(self.sector_etfs.keys()) + [self.market_etf]
            
            # Fetch historical data
            data = yf.download(
                symbols,
                period=f"{lookback_days+5}d",  # Add a few days buffer
                group_by='ticker',
                progress=False
            )
            
            # Process data
            sector_data = {}
            
            # Timeframes to analyze
            timeframes = {
                '1D': 1,
                '1W': 5,
                '1M': 21,
                '3M': 63,
                'Full': lookback_days
            }
            
            # Get market (S&P 500) performance as baseline
            if self.market_etf in data:
                market_data = data[self.market_etf]['Close']
                
                # Calculate market returns for each timeframe
                market_returns = {}
                for name, days in timeframes.items():
                    if len(market_data) > days:
                        market_returns[name] = (market_data.iloc[-1] / market_data.iloc[-days-1] - 1) * 100
                    else:
                        market_returns[name] = (market_data.iloc[-1] / market_data.iloc[0] - 1) * 100
                
                sector_data['market_returns'] = market_returns
            
            # Process each sector
            sectors_performance = []
            
            for symbol, sector_name in self.sector_etfs.items():
                if symbol in data:
                    sector_price = data[symbol]['Close']
                    sector_volume = data[symbol]['Volume']
                    
                    if len(sector_price) > 0:
                        # Calculate returns for different timeframes
                        returns = {}
                        for name, days in timeframes.items():
                            if len(sector_price) > days:
                                returns[name] = (sector_price.iloc[-1] / sector_price.iloc[-days-1] - 1) * 100
                            else:
                                returns[name] = (sector_price.iloc[-1] / sector_price.iloc[0] - 1) * 100
                        
                        # Calculate relative strength vs market
                        relative_strength = {}
                        for timeframe in returns.keys():
                            if timeframe in market_returns:
                                relative_strength[timeframe] = returns[timeframe] - market_returns[timeframe]
                            else:
                                relative_strength[timeframe] = 0.0
                        
                        # Calculate momentum (rate of change in relative strength)
                        momentum = relative_strength['1W'] - relative_strength['1M'] if '1W' in relative_strength and '1M' in relative_strength else 0
                        
                        # Calculate volume trend
                        recent_volume = sector_volume[-5:].mean()
                        previous_volume = sector_volume[-10:-5].mean()
                        volume_change = (recent_volume / previous_volume - 1) * 100 if previous_volume > 0 else 0
                        
                        # Current price
                        current_price = sector_price.iloc[-1]
                        
                        # Calculate simple moving averages
                        sma20 = sector_price[-20:].mean() if len(sector_price) >= 20 else sector_price.mean()
                        sma50 = sector_price[-50:].mean() if len(sector_price) >= 50 else sector_price.mean()
                        
                        # Determine trend based on SMAs
                        if current_price > sma20 and sma20 > sma50:
                            trend = "Strong Uptrend"
                        elif current_price > sma20:
                            trend = "Uptrend"
                        elif current_price < sma20 and sma20 < sma50:
                            trend = "Strong Downtrend"
                        elif current_price < sma20:
                            trend = "Downtrend"
                        else:
                            trend = "Neutral"
                        
                        # Add to results
                        sectors_performance.append({
                            'symbol': symbol,
                            'sector': sector_name,
                            'price': current_price,
                            'returns_1d': returns.get('1D', 0),
                            'returns_1w': returns.get('1W', 0),
                            'returns_1m': returns.get('1M', 0),
                            'returns_3m': returns.get('3M', 0),
                            'relative_strength_1d': relative_strength.get('1D', 0),
                            'relative_strength_1w': relative_strength.get('1W', 0),
                            'relative_strength_1m': relative_strength.get('1M', 0),
                            'relative_strength_3m': relative_strength.get('3M', 0),
                            'momentum': momentum,
                            'volume_change': volume_change,
                            'trend': trend
                        })
            
            # Sort by relative strength (1-week)
            sectors_performance.sort(key=lambda x: x['relative_strength_1w'], reverse=True)
            
            # Identify leading and lagging sectors
            leading_sectors = [s['sector'] for s in sectors_performance[:3]]
            lagging_sectors = [s['sector'] for s in sectors_performance[-3:]]
            
            # Determine market phase based on leading sectors
            # Simplified model:
            # - Early Bull: Financials, Consumer Discretionary leading
            # - Late Bull: Technology, Materials leading
            # - Early Bear: Utilities, Consumer Staples leading
            # - Late Bear: Energy, Healthcare, Financials showing relative strength
            
            market_phase = "Undetermined"
            
            if 'Financial' in leading_sectors or 'Consumer Discretionary' in leading_sectors:
                if 'Technology' in leading_sectors:
                    market_phase = "Late Bull Market"
                else:
                    market_phase = "Early Bull Market"
            
            elif 'Technology' in leading_sectors or 'Materials' in leading_sectors:
                market_phase = "Late Bull Market"
            
            elif 'Utilities' in leading_sectors or 'Consumer Staples' in leading_sectors:
                market_phase = "Early Bear Market" 
            
            elif 'Energy' in leading_sectors or 'Healthcare' in leading_sectors:
                if 'Financial' in leading_sectors:
                    market_phase = "Late Bear Market"
                else:
                    market_phase = "Bear Market"
            
            # Create sector summary
            sector_data['sectors'] = sectors_performance
            sector_data['leading_sectors'] = leading_sectors
            sector_data['lagging_sectors'] = lagging_sectors
            sector_data['market_phase'] = market_phase
            sector_data['analysis_date'] = datetime.now().isoformat()
            
            # Generate visualization
            self._generate_sector_heatmap(sectors_performance)
            
            # Save results to CSV
            df = pd.DataFrame(sectors_performance)
            filename = os.path.join(self.results_dir, f"sector_performance_{datetime.now().strftime('%Y%m%d')}.csv")
            df.to_csv(filename, index=False)
            logger.info(f"Saved sector performance data to {filename}")
            
            return sector_data
            
        except Exception as e:
            logger.error(f"Error analyzing sector performance: {e}")
            return {
                'sectors': [],
                'leading_sectors': [],
                'lagging_sectors': [],
                'market_phase': 'Unknown',
                'analysis_date': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_sector_for_symbol(self, symbol) -> str:
        """
        Get the sector for a given symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Sector name
        """
        try:
            import yfinance as yf
            
            # Try to get info from yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if 'sector' in info and info['sector']:
                return info['sector']
            
            # If that fails, check if it's a sector ETF
            if symbol in self.sector_etfs:
                return self.sector_etfs[symbol]
            
            # Default fallback
            return "Unknown"
            
        except Exception as e:
            logger.warning(f"Could not get sector for {symbol}: {e}")
            return "Unknown"
    
    def get_stocks_in_sector(self, sector_name) -> List[str]:
        """
        Get a list of stocks in a given sector
        
        Args:
            sector_name: Name of sector
            
        Returns:
            List of stock symbols in the sector
        """
        logger.info(f"Finding stocks in {sector_name} sector")
        
        try:
            # Try to get S&P 500 and filter by sector
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            sector_stocks = sp500[sp500['GICS Sector'] == sector_name]['Symbol'].tolist()
            
            if sector_stocks:
                logger.info(f"Found {len(sector_stocks)} stocks in {sector_name} sector")
                return sector_stocks
            else:
                logger.warning(f"No stocks found in {sector_name} sector")
                return []
                
        except Exception as e:
            logger.warning(f"Error getting stocks in {sector_name} sector: {e}")
            
            # Return some default stocks for common sectors
            default_stocks = {
                'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD'],
                'Financial': ['JPM', 'BAC', 'GS', 'MS', 'V'],
                'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
                'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABBV', 'UNH'],
                'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
                'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
                'Industrials': ['BA', 'HON', 'UNP', 'UPS', 'CAT'],
                'Materials': ['LIN', 'APD', 'ECL', 'DD', 'NEM'],
                'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP'],
                'Real Estate': ['AMT', 'PLD', 'CCI', 'PSA', 'EQIX'],
                'Communication Services': ['META', 'GOOG', 'T', 'VZ', 'CMCSA']
            }
            
            return default_stocks.get(sector_name, [])
    
    def find_strong_sectors(self) -> List[Dict[str, Any]]:
        """
        Find sectors showing relative strength
        
        Returns:
            List of strong sectors with potential trade ideas
        """
        logger.info("Finding strong sectors for potential trades")
        
        try:
            # Analyze sector performance
            sector_data = self.analyze_sector_performance(lookback_days=20)
            
            if not sector_data or 'sectors' not in sector_data or not sector_data['sectors']:
                logger.warning("No sector data available")
                return []
            
            # Find sectors with positive momentum and relative strength
            strong_sectors = []
            
            for sector in sector_data['sectors']:
                # Criteria for strong sector:
                # 1. Positive 1-week relative strength
                # 2. Positive momentum
                # 3. In an uptrend
                
                if (sector['relative_strength_1w'] > 0 and 
                        sector['momentum'] > 0 and 
                        'Uptrend' in sector['trend']):
                    
                    # Get some stocks in this sector
                    stocks = self.get_stocks_in_sector(sector['sector'])
                    
                    # Add to results
                    strong_sectors.append({
                        'sector': sector['sector'],
                        'etf': sector['symbol'],
                        'relative_strength': sector['relative_strength_1w'],
                        'momentum': sector['momentum'],
                        'trend': sector['trend'],
                        'key_stocks': stocks[:5]  # Just include top 5
                    })
            
            # Sort by relative strength
            strong_sectors.sort(key=lambda x: x['relative_strength'], reverse=True)
            
            logger.info(f"Found {len(strong_sectors)} strong sectors")
            return strong_sectors
            
        except Exception as e:
            logger.error(f"Error finding strong sectors: {e}")
            return []
    
    def _generate_sector_heatmap(self, sectors_performance):
        """Generate a heatmap of sector performance"""
        try:
            if not sectors_performance:
                return
                
            # Create a DataFrame for the heatmap
            heatmap_data = []
            for sector in sectors_performance:
                heatmap_data.append({
                    'Sector': sector['sector'],
                    '1 Day': sector['returns_1d'],
                    '1 Week': sector['returns_1w'],
                    '1 Month': sector['returns_1m'],
                    '3 Month': sector['returns_3m']
                })
            
            df = pd.DataFrame(heatmap_data)
            df.set_index('Sector', inplace=True)
            
            # Create the heatmap
            plt.figure(figsize=(12, 8))
            
            # Custom color map for returns (red for negative, green for positive)
            cmap = sns.diverging_palette(10, 120, as_cmap=True)
            
            # Plot the heatmap
            sns.heatmap(df, annot=True, cmap=cmap, center=0, fmt='.2f',
                        linewidths=.5, cbar_kws={'label': 'Returns (%)'})
            
            plt.title('Sector Performance Heatmap (%)', fontsize=16)
            plt.tight_layout()
            
            # Save the figure
            filepath = os.path.join(self.results_dir, f"sector_heatmap_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved sector heatmap to {filepath}")
            
        except Exception as e:
            logger.error(f"Error generating sector heatmap: {e}") 