import pandas as pd
import numpy as np
import logging
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class NewsIntegration:
    """
    Integrates news and catalysts into trading decisions
    """
    
    def __init__(self):
        """Initialize news integration"""
        self.news_cache = {}
        self.cache_expiry = {}
        self.news_dir = 'results/news'
        os.makedirs(self.news_dir, exist_ok=True)
        
        # Define API keys (these would be set via configuration in a real app)
        self.api_keys = {
            'news_api': os.environ.get('NEWS_API_KEY', ''),
            'alpha_vantage': os.environ.get('ALPHA_VANTAGE_KEY', '')
        }
    
    def get_stock_news(self, symbol: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent news for a specific stock
        
        Args:
            symbol: Stock symbol
            max_results: Maximum number of news items to return
            
        Returns:
            List of news items with title, source, date, url, and sentiment
        """
        logger.info(f"Getting news for {symbol}")
        
        # Check cache first (with 15 minute expiry)
        cache_key = f"{symbol}_news"
        if cache_key in self.news_cache and cache_key in self.cache_expiry:
            if datetime.now() < self.cache_expiry[cache_key]:
                return self.news_cache[cache_key][:max_results]
        
        # Collect news from multiple sources
        news_items = []
        
        # Try Alpha Vantage if key is available
        if self.api_keys['alpha_vantage']:
            try:
                av_news = self._get_alpha_vantage_news(symbol)
                if av_news:
                    news_items.extend(av_news)
            except Exception as e:
                logger.warning(f"Error getting Alpha Vantage news for {symbol}: {e}")
        
        # Try Yahoo Finance (scraping)
        try:
            yahoo_news = self._get_yahoo_finance_news(symbol)
            if yahoo_news:
                news_items.extend(yahoo_news)
        except Exception as e:
            logger.warning(f"Error getting Yahoo Finance news for {symbol}: {e}")
        
        # Add sentiment analysis
        news_items = self._analyze_news_sentiment(news_items)
        
        # Sort by date (newest first)
        news_items.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        # Deduplicate news
        unique_news = []
        seen_titles = set()
        for item in news_items:
            # Use normalized title to detect duplicates
            norm_title = self._normalize_title(item.get('title', ''))
            if norm_title and norm_title not in seen_titles:
                seen_titles.add(norm_title)
                unique_news.append(item)
        
        # Cache results for 15 minutes
        self.news_cache[cache_key] = unique_news
        self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=15)
        
        # Save to file for reference
        self._save_news_to_file(symbol, unique_news)
        
        return unique_news[:max_results]
    
    def get_market_news(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent market news
        
        Args:
            max_results: Maximum number of news items to return
            
        Returns:
            List of market news items
        """
        logger.info("Getting market news")
        
        # Check cache (with 30 minute expiry)
        cache_key = "market_news"
        if cache_key in self.news_cache and cache_key in self.cache_expiry:
            if datetime.now() < self.cache_expiry[cache_key]:
                return self.news_cache[cache_key][:max_results]
        
        # Collect news from multiple sources
        news_items = []
        
        # Try Market news from Alpha Vantage
        if self.api_keys['alpha_vantage']:
            try:
                av_news = self._get_alpha_vantage_news("SPY")  # Use SPY as proxy for market
                if av_news:
                    news_items.extend(av_news)
            except Exception as e:
                logger.warning(f"Error getting Alpha Vantage market news: {e}")
        
        # Try CNBC (scraping)
        try:
            cnbc_news = self._get_cnbc_market_news()
            if cnbc_news:
                news_items.extend(cnbc_news)
        except Exception as e:
            logger.warning(f"Error getting CNBC market news: {e}")
        
        # Add sentiment analysis
        news_items = self._analyze_news_sentiment(news_items)
        
        # Sort by date (newest first)
        news_items.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        # Deduplicate news
        unique_news = []
        seen_titles = set()
        for item in news_items:
            norm_title = self._normalize_title(item.get('title', ''))
            if norm_title and norm_title not in seen_titles:
                seen_titles.add(norm_title)
                unique_news.append(item)
        
        # Cache results for 30 minutes
        self.news_cache[cache_key] = unique_news
        self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=30)
        
        # Save to file for reference
        self._save_news_to_file("MARKET", unique_news)
        
        return unique_news[:max_results]
    
    def find_catalyst_stocks(self) -> List[Dict[str, Any]]:
        """
        Find stocks with significant news catalysts
        
        Returns:
            List of stocks with recent catalysts
        """
        logger.info("Scanning for stocks with catalysts")
        
        # Start with common volatile stocks
        symbols = [
            "TSLA", "NVDA", "AMD", "AAPL", "MSFT", "AMZN", "META", "GOOGL", 
            "NFLX", "BA", "PLTR", "COIN", "GME", "AMC", "BBBY", "HOOD", "RIVN"
        ]
        
        catalyst_stocks = []
        
        for symbol in symbols:
            try:
                # Get news for the symbol
                news = self.get_stock_news(symbol, max_results=3)
                
                if news:
                    # Check for catalysts in news headlines
                    catalyst_words = [
                        'upgrade', 'downgrade', 'beats', 'misses', 'raises', 'lowers', 
                        'guidance', 'announces', 'launches', 'approves', 'rejects', 
                        'acquires', 'merger', 'investigation', 'lawsuit', 'settlement',
                        'exclusive', 'breaking', 'reports', 'exceeds', 'falls short'
                    ]
                    
                    catalysts = []
                    high_sentiment = False
                    
                    for item in news:
                        title = item.get('title', '').lower()
                        
                        # Check for catalyst words
                        for word in catalyst_words:
                            if word in title:
                                catalysts.append(item)
                                break
                        
                        # Check for strong sentiment
                        sentiment = item.get('sentiment', 0)
                        if abs(sentiment) > 0.3:  # Strong positive or negative
                            high_sentiment = True
                            catalysts.append(item)
                    
                    # If catalysts found, add to results
                    if catalysts:
                        catalyst_stocks.append({
                            'symbol': symbol,
                            'catalysts': catalysts[:3],  # Limit to top 3
                            'high_sentiment': high_sentiment,
                            'catalyst_count': len(catalysts)
                        })
                        
                        logger.info(f"Found {len(catalysts)} catalysts for {symbol}")
            
            except Exception as e:
                logger.warning(f"Error scanning {symbol} for catalysts: {e}")
        
        # Sort by number of catalysts
        catalyst_stocks.sort(key=lambda x: x['catalyst_count'], reverse=True)
        
        return catalyst_stocks
    
    def _get_alpha_vantage_news(self, symbol) -> List[Dict[str, Any]]:
        """Get news from Alpha Vantage API"""
        news = []
        
        if not self.api_keys['alpha_vantage']:
            return news
            
        try:
            # Set up the API request
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "apikey": self.api_keys['alpha_vantage'],
                "limit": 10
            }
            
            # Make the request
            response = requests.get(url, params=params)
            data = response.json()
            
            # Process the response
            if 'feed' in data:
                for item in data['feed']:
                    # Convert time to datetime
                    try:
                        date = datetime.fromisoformat(item.get('time_published', '').replace('Z', '+00:00'))
                    except:
                        date = datetime.now()
                    
                    news.append({
                        'title': item.get('title', ''),
                        'summary': item.get('summary', ''),
                        'source': item.get('source', ''),
                        'url': item.get('url', ''),
                        'date': date,
                        'raw_sentiment': item.get('overall_sentiment_score', 0)
                    })
            
            return news
            
        except Exception as e:
            logger.error(f"Error in Alpha Vantage news API: {e}")
            return news
    
    def _get_yahoo_finance_news(self, symbol) -> List[Dict[str, Any]]:
        """Get news from Yahoo Finance (web scraping)"""
        news = []
        
        try:
            # Set up the request
            url = f"https://finance.yahoo.com/quote/{symbol}/news"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Make the request
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news items
            news_items = soup.find_all('li', class_='js-stream-content')
            
            for item in news_items:
                # Extract the headline
                headline = item.find('h3')
                if headline:
                    title = headline.text.strip()
                else:
                    continue
                
                # Extract the link
                link = item.find('a', href=True)
                if link:
                    url = link['href']
                    if not url.startswith('http'):
                        url = f"https://finance.yahoo.com{url}"
                else:
                    url = ''
                
                # Extract the source and date
                source_element = item.find('div', class_='C(#959595)')
                source = ''
                date_str = ''
                
                if source_element:
                    source_text = source_element.text.strip()
                    parts = source_text.split('·')
                    
                    if len(parts) >= 1:
                        source = parts[0].strip()
                    
                    if len(parts) >= 2:
                        date_str = parts[1].strip()
                
                # Convert relative date to actual date
                date = self._parse_relative_date(date_str)
                
                # Add to news list
                news.append({
                    'title': title,
                    'source': source,
                    'url': url,
                    'date': date,
                    'summary': ''
                })
            
            return news
            
        except Exception as e:
            logger.error(f"Error scraping Yahoo Finance news: {e}")
            return news
    
    def _get_cnbc_market_news(self) -> List[Dict[str, Any]]:
        """Get market news from CNBC (web scraping)"""
        news = []
        
        try:
            # Set up the request
            url = "https://www.cnbc.com/markets/"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Make the request
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news cards
            news_cards = soup.find_all('div', class_='Card-titleContainer')
            
            for card in news_cards:
                # Extract the headline
                headline = card.find('a', class_='Card-title')
                if headline:
                    title = headline.text.strip()
                else:
                    continue
                
                # Extract the link
                if headline and 'href' in headline.attrs:
                    url = headline['href']
                else:
                    url = ''
                
                # Extract the time
                time_element = card.find('span', class_='Card-time')
                date_str = time_element.text.strip() if time_element else ''
                
                # Convert relative date to actual date
                date = self._parse_relative_date(date_str)
                
                # Add to news list
                news.append({
                    'title': title,
                    'source': 'CNBC',
                    'url': url,
                    'date': date,
                    'summary': ''
                })
            
            return news
            
        except Exception as e:
            logger.error(f"Error scraping CNBC market news: {e}")
            return news
    
    def _analyze_news_sentiment(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add sentiment analysis to news items
        
        Args:
            news_items: List of news items to analyze
            
        Returns:
            News items with sentiment scores added
        """
        # If raw_sentiment is already available, use it
        for item in news_items:
            if 'raw_sentiment' in item:
                item['sentiment'] = item['raw_sentiment']
            else:
                # Simple rule-based sentiment analysis
                sentiment = self._analyze_text_sentiment(item.get('title', '') + ' ' + item.get('summary', ''))
                item['sentiment'] = sentiment
        
        return news_items
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """
        Perform simple rule-based sentiment analysis
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        # Positive and negative word lists
        positive_words = [
            'up', 'rise', 'rising', 'gain', 'gains', 'positive', 'profit', 'profits', 
            'beat', 'beats', 'exceeds', 'higher', 'rally', 'bullish', 'growth', 
            'outperform', 'upgrade', 'strong', 'success', 'successful', 'approve',
            'approved', 'win', 'wins', 'good', 'great', 'excellent', 'breakthrough'
        ]
        
        negative_words = [
            'down', 'fall', 'falling', 'lose', 'loss', 'losses', 'negative', 'miss', 
            'misses', 'lower', 'bearish', 'decline', 'weaker', 'downgrade', 'weak',
            'fails', 'failed', 'disappointing', 'disappoints', 'investigation', 
            'lawsuit', 'penalty', 'fined', 'bad', 'poor', 'trouble', 'warning'
        ]
        
        # Normalize text
        text = text.lower()
        
        # Count occurrences
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        # Calculate sentiment score
        if positive_count == 0 and negative_count == 0:
            return 0.0
        
        total = positive_count + negative_count
        sentiment = (positive_count - negative_count) / total
        
        return sentiment
    
    def _parse_relative_date(self, date_str: str) -> datetime:
        """
        Parse relative date strings like "3 hours ago" into datetime objects
        
        Args:
            date_str: Relative date string
            
        Returns:
            Datetime object
        """
        now = datetime.now()
        
        # Handle empty strings
        if not date_str:
            return now
        
        # Try to match patterns like "X hours ago", "yesterday", etc.
        if 'minute' in date_str.lower():
            match = re.search(r'(\d+)', date_str)
            if match:
                minutes = int(match.group(1))
                return now - timedelta(minutes=minutes)
        
        elif 'hour' in date_str.lower():
            match = re.search(r'(\d+)', date_str)
            if match:
                hours = int(match.group(1))
                return now - timedelta(hours=hours)
        
        elif 'day' in date_str.lower() or 'yesterday' in date_str.lower():
            match = re.search(r'(\d+)', date_str)
            if match:
                days = int(match.group(1))
                return now - timedelta(days=days)
            elif 'yesterday' in date_str.lower():
                return now - timedelta(days=1)
        
        elif 'week' in date_str.lower():
            match = re.search(r'(\d+)', date_str)
            if match:
                weeks = int(match.group(1))
                return now - timedelta(weeks=weeks)
        
        # Try to parse absolute date
        try:
            return pd.to_datetime(date_str).to_pydatetime()
        except:
            # Default to current time if all else fails
            return now
    
    def _normalize_title(self, title: str) -> str:
        """
        Normalize a title for deduplication
        
        Args:
            title: News title
            
        Returns:
            Normalized title
        """
        # Remove punctuation, lowercase, etc.
        return re.sub(r'[^\w\s]', '', title.lower())
    
    def _save_news_to_file(self, symbol: str, news_items: List[Dict[str, Any]]) -> None:
        """
        Save news items to a CSV file
        
        Args:
            symbol: Stock symbol or 'MARKET'
            news_items: List of news items
        """
        if not news_items:
            return
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame(news_items)
            
            # Format dates
            if 'date' in df.columns:
                df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, datetime) else x)
            
            # Save to CSV
            filename = os.path.join(self.news_dir, f"{symbol}_news_{datetime.now().strftime('%Y%m%d')}.csv")
            df.to_csv(filename, index=False)
            logger.debug(f"Saved {len(news_items)} news items for {symbol} to {filename}")
        except Exception as e:
            logger.warning(f"Error saving news to file: {e}")
    
    def get_stock_news_with_price_impact(self, symbol: str, price_data=None) -> List[Dict[str, Any]]:
        """
        Get news with potential price impact analysis
        
        Args:
            symbol: Stock symbol
            price_data: Optional price data to correlate with news
            
        Returns:
            News items with price impact analysis
        """
        news = self.get_stock_news(symbol)
        
        # If no price data provided, we can't analyze impact
        if price_data is None or news is None:
            return news
        
        try:
            # Convert price_data index to datetime if needed
            if not isinstance(price_data.index, pd.DatetimeIndex):
                price_data.index = pd.to_datetime(price_data.index)
            
            # Analyze each news item for price impact
            for item in news:
                news_date = item.get('date')
                if not isinstance(news_date, datetime):
                    continue
                
                # Find price data closest to news release
                closest_date = price_data.index[price_data.index.get_indexer([news_date], method='nearest')[0]]
                
                # Get prices before and after news
                try:
                    # Get index position
                    date_loc = price_data.index.get_loc(closest_date)
                    
                    # Get prices 1 day before and after if available
                    before_idx = max(0, date_loc - 1)
                    after_idx = min(len(price_data) - 1, date_loc + 1)
                    
                    before_price = price_data['close'].iloc[before_idx]
                    after_price = price_data['close'].iloc[after_idx]
                    
                    # Calculate price change
                    price_change_pct = ((after_price - before_price) / before_price) * 100
                    
                    # Add to news item
                    item['price_impact_pct'] = price_change_pct
                    
                    # Determine if news sentiment matches price movement
                    sentiment = item.get('sentiment', 0)
                    sentiment_matches = (sentiment > 0 and price_change_pct > 0) or (sentiment < 0 and price_change_pct < 0)
                    
                    item['sentiment_matches_price'] = sentiment_matches
                    
                except Exception as e:
                    logger.debug(f"Error calculating price impact: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Error analyzing price impact: {e}")
        
        return news 