# main.py updates
from backtesting.backtest import Backtest
from models.model_trainer import train_model
import datetime
import logging
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import json
import joblib

try:
    from trading.risk_management import RiskManager, PositionSizing
except ImportError:
    logging.warning("Could not import risk_management module, using placeholder")
    
    class RiskManager:
        def __init__(self, *args, **kwargs):
            pass
            
        def validate_trade(self, trade_symbol, trade_size, trade_direction, current_portfolio, historical_data, sector_data):
            return True, "No risk validation performed"

try:
    from trading.regime_detector import RegimeDetector
except ImportError:
    logging.warning("Could not import regime_detector module, using placeholder")
    
    class RegimeDetector:
        def __init__(self):
            self.current_regime = "neutral"
            
        def detect_regime(self, market_data=None):
            return "neutral"

try:
    from trading.strategy_engine import StrategyEngine
except ImportError:
    logging.warning("Could not import strategy_engine module, using placeholder")
    
    class StrategyEngine:
        def __init__(self, *args, **kwargs):
            pass
            
        def generate_suggestions(self, historical_data=None, portfolio=None, market_regime="neutral"):
            return []

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("smart_trade_advisor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("smart_trade_advisor")

# Try importing modules, with fallbacks if they're not available
try:
    from backtesting.backtest import Backtest
except ImportError as e:
    logger.warning(f"Could not import backtesting module: {e}")
    # Define a simple placeholder
    class Backtest:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, *args, **kwargs):
            return {"returns": [], "metrics": {"sharpe_ratio": 0, "max_drawdown": 0}}

# Import other core components with similar try/except blocks
try:
    from data.historical_data_fetcher import HistoricalDataFetcher
    from data.realtime_data_stream import RealtimeDataStream
except ImportError:
    logger.warning("Could not import data modules, using placeholders")
    # Define placeholder classes

class HistoricalDataFetcher:
    def fetch_daily_data(self, symbol, days=180):
        logger.info(f"Placeholder: Fetching data for {symbol} for {days} days")
        return None

class RealtimeDataStream:
    def get_last_price(self, symbol):
        # Return dummy prices for demo
        prices = {
            "AAPL": 175.50,
            "MSFT": 320.75,
            "GOOGL": 2750.25,
            "NVDA": 850.30
        }
        return prices.get(symbol, 100.0)
    
    def disconnect(self):
        pass
    
    def subscribe_symbols(self, symbols):
        pass

class SmartTradeAdvisor:
    """
    Main application class that orchestrates the trading system
    """
    
    def __init__(self, config_path: str = "config.json"):
        logger.info("Initializing Smart Trade Advisor...")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize core components with try/except to handle import errors
        try:
            self.data_fetcher = HistoricalDataFetcher()
            self.realtime_stream = RealtimeDataStream()
        except Exception as e:
            logger.error(f"Error initializing data components: {e}")
            self.data_fetcher = HistoricalDataFetcher()
            self.realtime_stream = RealtimeDataStream()
        
        # System state
        self.is_running = False
        self.current_portfolio = self._load_portfolio("sample_portfolio.json")
        self.trade_suggestions = []
        self.current_regime = "neutral"  # Default regime
        
        # Add strategy engine
        self.strategy_engine = StrategyEngine()
        self.regime_detector = RegimeDetector()
        
        logger.info("Smart Trade Advisor initialized successfully")
    
    def detect_market_regime(self):
        """Detect the current market regime"""
        try:
            self.current_regime = self.regime_detector.detect_regime()
            logger.info(f"Current market regime: {self.current_regime}")
            return self.current_regime
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return "neutral"  # Default regime
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {
                "risk_parameters": {
                    "max_portfolio_risk": 0.02,
                    "max_position_size": 0.15,
                    "max_sector_exposure": 0.25,
                },
                "trading_parameters": {
                    "strategies": ["trend_following", "mean_reversion"],
                    "confidence_threshold": 0.65,
                }
            }
    
    def _load_portfolio(self, portfolio_file: str) -> Dict[str, Any]:
        """Load portfolio from file"""
        try:
            with open(portfolio_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"Could not load portfolio: {portfolio_file}. Using default.")
            return {
                "cash": 100000,
                "positions": {
                    "AAPL": {"shares": 100, "average_price": 150.0},
                    "MSFT": {"shares": 50, "average_price": 280.0},
                    "GOOGL": {"shares": 25, "average_price": 2500.0}
                },
                "portfolio_id": "demo_portfolio_001"
            }
            
    def start(self) -> None:
        """Start the trading advisor system"""
        if self.is_running:
            logger.warning("System is already running")
            return
            
        logger.info("Starting Smart Trade Advisor system")
        self.is_running = True
        logger.info("Smart Trade Advisor started successfully!")
    
    def stop(self) -> None:
        """Stop the trading advisor system"""
        if not self.is_running:
            logger.warning("System is already stopped")
            return
            
        logger.info("Stopping Smart Trade Advisor system")
        self.is_running = False
        logger.info("Smart Trade Advisor stopped successfully")
    
    def generate_trade_suggestions(self) -> List[Dict[str, Any]]:
        """
        Generate trade suggestions based on portfolio and market conditions using trained models
        
        Returns:
            List of trade suggestions
        """
        logger.info("Generating trade suggestions using trained models...")
        
        # Get current market regime
        market_regime = self.current_regime or "neutral"
        
        # Get historical data for stocks in portfolio and watchlist
        portfolio_symbols = list(self.current_portfolio.get("positions", {}).keys())
        watchlist_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]
        
        all_symbols = list(set(portfolio_symbols + watchlist_symbols))
        
        try:
            # Step 1: Load the trained models
            models_dir = 'models/trained'
            if not os.path.exists(models_dir):
                logger.warning("No trained models found. Please run model training first.")
                raise FileNotFoundError("No trained models directory")
            
            # Load models for each symbol
            trained_models = {}
            for file in os.listdir(models_dir):
                if file.endswith('.joblib'):
                    try:
                        symbol = file.split('_')[0]
                        if symbol in all_symbols:
                            model_path = os.path.join(models_dir, file)
                            trained_models[symbol] = joblib.load(model_path)
                            logger.info(f"Loaded model for {symbol}")
                    except Exception as e:
                        logger.error(f"Error loading model {file}: {e}")
            
            # Step 2: Get historical data for feature engineering
            historical_data = {}
            for symbol in all_symbols:
                try:
                    # Try to get real data
                    data = self.data_fetcher.fetch_daily_data(symbol, days=60)
                    if data is not None and not data.empty:
                        historical_data[symbol] = data
                    else:
                        raise ValueError(f"No data returned for {symbol}")
                except Exception as e:
                    logger.warning(f"Error fetching data for {symbol}: {e}")
                    
                    # Generate mock data if real data fetch fails
                    dates = pd.date_range(end=datetime.datetime.now(), periods=60)
                    mock_data = pd.DataFrame({
                        'date': dates,
                        'open': np.random.normal(100, 5, 60) * (1 + 0.1 * np.sin(np.linspace(0, 4*np.pi, 60))),
                        'high': np.random.normal(102, 5, 60) * (1 + 0.1 * np.sin(np.linspace(0, 4*np.pi, 60))),
                        'low': np.random.normal(98, 5, 60) * (1 + 0.1 * np.sin(np.linspace(0, 4*np.pi, 60))),
                        'close': np.random.normal(100, 5, 60) * (1 + 0.1 * np.sin(np.linspace(0, 4*np.pi, 60))),
                        'volume': np.random.normal(1000000, 200000, 60) * (1 + 0.1 * np.sin(np.linspace(0, 4*np.pi, 60)))
                    })
                    mock_data.set_index('date', inplace=True)
                    historical_data[symbol] = mock_data
            
            # Step 3: Engineer features for prediction
            # Create a ModelTrainer instance to use its feature engineering methods
            from models.model_trainer import ModelTrainer
            model_trainer = ModelTrainer()
            
            # Generate predictions for each symbol
            predictions = {}
            for symbol in all_symbols:
                if symbol in historical_data and symbol in trained_models:
                    # Get the data
                    symbol_data = historical_data[symbol].copy()
                    
                    # Engineer features
                    feature_data = model_trainer._engineer_features(
                        symbol_data,
                        ['rsi', 'macd', 'bollinger', 'volume', 'momentum', 'trend'],
                        30  # lookback period
                    )
                    
                    # Select feature columns (exclude target, open, high, low, close, volume)
                    feature_cols = [col for col in feature_data.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
                    
                    # Drop NaN values
                    features = feature_data[feature_cols].dropna()
                    
                    if len(features) > 0:
                        # Scale features
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        features_scaled = scaler.fit_transform(features)
                        
                        # Make prediction
                        model = trained_models[symbol]
                        prediction = model.predict(features_scaled[-1:])  # Predict using the most recent data point
                        
                        # Store prediction
                        predictions[symbol] = {
                            'predicted_return': prediction[0],
                            'current_price': self.realtime_stream.get_last_price(symbol),
                            'prediction_date': datetime.datetime.now().isoformat()
                        }
                        
                        logger.info(f"Predicted return for {symbol}: {prediction[0]:.2%}")
            
            # Step 4: Generate trade suggestions based on predictions
            suggestions = []
            
            # Sort symbols by predicted return (descending)
            sorted_symbols = sorted(
                predictions.keys(),
                key=lambda symbol: predictions[symbol]['predicted_return'],
                reverse=True
            )
            
            # Generate buy suggestions for top performers
            for symbol in sorted_symbols[:3]:  # Top 3 symbols
                if predictions[symbol]['predicted_return'] > 0.005:  # Only suggest if predicted return > 0.5%
                    # Calculate quantity based on portfolio value and position sizing
                    portfolio_value = self.current_portfolio.get("cash", 0)
                    for s, pos in self.current_portfolio.get("positions", {}).items():
                        portfolio_value += pos.get("shares", 0) * self.realtime_stream.get_last_price(s)
                    
                    # Position size: 5% of portfolio per trade
                    position_value = portfolio_value * 0.05
                    price = predictions[symbol]['current_price']
                    quantity = int(position_value / price) if price > 0 else 0
                    
                    if quantity > 0:
                        suggestions.append({
                            "symbol": symbol,
                            "action": "BUY",
                            "quantity": quantity,
                            "reason": f"ML model predicts {predictions[symbol]['predicted_return']:.2%} return",
                            "confidence": min(0.5 + abs(predictions[symbol]['predicted_return']) * 50, 0.95),  # Scale confidence
                            "strategy": "Machine Learning Prediction" if market_regime == "neutral" 
                                       else f"ML Prediction ({market_regime} regime)"
                        })
            
            # Generate sell suggestions for worst performers
            for symbol in reversed(sorted_symbols[-3:]):  # Bottom 3 symbols
                if predictions[symbol]['predicted_return'] < -0.005 and symbol in portfolio_symbols:  # Only sell if we own it
                    # Get current position
                    position = self.current_portfolio.get("positions", {}).get(symbol, {})
                    shares = position.get("shares", 0)
                    
                    if shares > 0:
                        # Sell half the position if the prediction is negative
                        quantity = max(1, int(shares * 0.5))
                        
                        suggestions.append({
                            "symbol": symbol,
                            "action": "SELL",
                            "quantity": quantity,
                            "reason": f"ML model predicts {predictions[symbol]['predicted_return']:.2%} return",
                            "confidence": min(0.5 + abs(predictions[symbol]['predicted_return']) * 50, 0.95),
                            "strategy": "Machine Learning Prediction" if market_regime == "neutral" 
                                       else f"ML Prediction ({market_regime} regime)"
                        })
            
            # Step 5: Validate trades against risk management rules
            valid_suggestions = []
            for suggestion in suggestions:
                # Calculate trade size
                trade_size = suggestion.get("quantity", 0) * (
                    self.realtime_stream.get_last_price(suggestion["symbol"]) or 100
                )
                
                # Use placeholder data for validation
                is_valid, reason = self.risk_manager.validate_trade(
                    trade_symbol=suggestion["symbol"],
                    trade_size=trade_size,
                    trade_direction=suggestion["action"],
                    current_portfolio=self.current_portfolio.get("positions", {}),
                    historical_data={},  # Would use actual historical data in practice
                    sector_data={}       # Would use actual sector data in practice
                )
                
                if is_valid:
                    valid_suggestions.append(suggestion)
                    logger.info(f"Generated valid trade suggestion: {suggestion['action']} {suggestion['quantity']} {suggestion['symbol']}")
                else:
                    logger.warning(f"Trade suggestion rejected by risk management: {reason}")
            
            self.trade_suggestions = valid_suggestions
            return valid_suggestions
            
        except Exception as e:
            logger.error(f"Error generating trade suggestions with ML models: {e}")
            logger.info("Falling back to strategy engine for suggestions")
            
            # Fall back to strategy engine if ML predictions fail
            try:
                suggestions = self.strategy_engine.generate_suggestions(
                    historical_data=historical_data if 'historical_data' in locals() else {},
                    portfolio=self.current_portfolio,
                    market_regime=market_regime
                )
                
                # Validate trades against risk management rules
                valid_suggestions = []
                for suggestion in suggestions:
                    # Calculate trade size
                    trade_size = suggestion.get("quantity", 0) * (
                        self.realtime_stream.get_last_price(suggestion["symbol"]) or 100
                    )
                    
                    # Use placeholder data for validation
                    is_valid, reason = self.risk_manager.validate_trade(
                        trade_symbol=suggestion["symbol"],
                        trade_size=trade_size,
                        trade_direction=suggestion["action"],
                        current_portfolio=self.current_portfolio.get("positions", {}),
                        historical_data={},
                        sector_data={}
                    )
                    
                    if is_valid:
                        valid_suggestions.append(suggestion)
                        logger.info(f"Generated valid trade suggestion: {suggestion['action']} {suggestion['quantity']} {suggestion['symbol']}")
                    else:
                        logger.warning(f"Trade suggestion rejected by risk management: {reason}")
                
                self.trade_suggestions = valid_suggestions
                return valid_suggestions
                
            except Exception as e2:
                logger.error(f"Error with strategy engine fallback: {e2}")
                # Fallback to sample suggestions if all else fails
                suggestions = [
                    {
                        "symbol": "AAPL",
                        "action": "SELL",
                        "quantity": 25,
                        "reason": "Fallback suggestion - technical indicators",
                        "confidence": 0.6,
                        "strategy": "Fallback Strategy"
                    },
                    {
                        "symbol": "NVDA",
                        "action": "BUY",
                        "quantity": 10,
                        "reason": "Fallback suggestion - momentum indicators",
                        "confidence": 0.65,
                        "strategy": "Fallback Strategy"
                    }
                ]
                
                self.trade_suggestions = suggestions
                return suggestions
    
    def execute_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade (simulated in this demonstration)
        
        Args:
            trade: Trade details
            
        Returns:
            Execution results
        """
        logger.info(f"Executing trade: {trade['action']} {trade['quantity']} {trade['symbol']}")
        
        # Simulate execution
        execution_price = self.realtime_stream.get_last_price(trade["symbol"])
        
        execution_result = {
            "symbol": trade["symbol"],
            "action": trade["action"],
            "quantity": trade["quantity"],
            "requested_price": execution_price,
            "executed_price": execution_price,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "FILLED",
            "trade_id": f"T{int(datetime.datetime.now().timestamp())}"
        }
        
        logger.info(f"Trade executed successfully: {execution_result['trade_id']}")
        return execution_result

    def run_analysis_cycle(self) -> List[Dict[str, Any]]:
        """
        Run a full analysis cycle and generate trade suggestions
        
        Returns:
            List of trade suggestions
        """
        if not self.is_running:
            logger.warning("Cannot run analysis cycle - system not started")
            return []
        
        # Update market data
        try:
            logger.info("Updating market data...")
            portfolio_symbols = list(self.current_portfolio.get("positions", {}).keys())
            self.realtime_stream.subscribe_symbols(portfolio_symbols)
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
        
        # Detect market regime
        self.detect_market_regime()
        
        # Generate trade suggestions
        suggestions = self.generate_trade_suggestions()
        
        return suggestions

    def check_ml_dependencies(self) -> bool:
        """
        Check if all required dependencies for machine learning are installed
        
        Returns:
            True if all dependencies are available, False otherwise
        """
        try:
            import joblib
            import sklearn
            import xgboost
            import matplotlib
            return True
        except ImportError as e:
            logger.warning(f"Missing ML dependency: {e}")
            logger.info("Please run install-ml-packages.bat to install required packages")
            return False

    def train_models(self, symbols=None, model_type='xgboost') -> Dict:
        """
        Train machine learning models for the specified symbols
        
        Args:
            symbols: List of symbols to train models for (None = use portfolio + watchlist)
            model_type: Type of model to train ('xgboost', 'random_forest', etc.)
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {model_type} models...")
        
        # Check dependencies
        if not self.check_ml_dependencies():
            logger.error("Required dependencies not installed. Cannot train models.")
            return {"status": "failed", "error": "Missing dependencies"}
        
        # Get symbols from portfolio and watchlist if not specified
        if symbols is None:
            portfolio_symbols = list(self.current_portfolio.get("positions", {}).keys())
            watchlist_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]
            symbols = list(set(portfolio_symbols + watchlist_symbols))
        
        # Prepare training configuration
        config = {
            'model_type': model_type,
            'symbols': symbols,
            'prediction_target': 'next_day_return',
            'prediction_horizon': 1,
            'features': ['rsi', 'macd', 'bollinger', 'volume', 'momentum', 'trend'],
            'lookback_period': 30,
            'test_size': 0.2,
            'cv_folds': 3,
            'run_backtest': True
        }
        
        # Set resource limits for training
        if model_type == 'xgboost':
            try:
                import xgboost as xgb
                # Check if we can load xgboost with no errors
            except ImportError as e:
                logger.error(f"XGBoost import error: {e}")
                logger.info("Falling back to random_forest model")
                model_type = 'random_forest'
            except Exception as e:
                logger.error(f"XGBoost error: {e}")
                logger.info("Falling back to random_forest model")
                model_type = 'random_forest'
                
        # Limit resources for lighter training
        resource_limits = {
            'n_estimators': 50,  # Fewer trees
            'threads': 2,        # Limit CPU usage
            'subsample': 0.8     # Use subset of data
        }
                
        try:
            from models.model_trainer import train_model
            trained_model = train_model(config)
            
            logger.info("Model training complete!")
            
            # Print performance summary
            performance_summary = {}
            for symbol, perf in trained_model['performance'].items():
                performance_summary[symbol] = {
                    'direction_accuracy': perf.get('direction_accuracy', 0),
                    'profit_factor': perf.get('profit_factor', 0)
                }
                logger.info(f"{symbol} Performance:")
                logger.info(f"  Direction Accuracy: {perf.get('direction_accuracy', 0):.2%}")
                logger.info(f"  Profit Factor: {perf.get('profit_factor', 0):.2f}")
            
            return {
                "status": "success", 
                "trained_symbols": list(trained_model['performance'].keys()),
                "performance": performance_summary
            }
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {"status": "failed", "error": str(e)}

    def get_top_movers(self, n=20, min_price=5.0, min_volume=500000) -> List[Dict[str, Any]]:
        """
        Identify top gaining and declining stocks in the market today
        
        Args:
            n: Number of top movers to return (total will be 2*n - gainers and losers)
            min_price: Minimum price filter to avoid penny stocks
            min_volume: Minimum volume filter to ensure liquidity
            
        Returns:
            List of dictionaries with top movers information
        """
        logger.info(f"Scanning for top {n} market movers...")
        
        try:
            # Try to use yfinance to get real market data
            import yfinance as yf
            
            # Get major index components to scan
            try:
                # S&P 500 components
                sp500 = yf.Ticker("^GSPC")
                sp500_holdings = list(pd.DataFrame(sp500.history_metadata.get('components', [])).columns)
                
                # NASDAQ 100 components
                nasdaq100 = yf.Ticker("^NDX") 
                nasdaq_holdings = list(pd.DataFrame(nasdaq100.history_metadata.get('components', [])).columns)
                
                # Combine unique symbols
                scan_symbols = list(set(sp500_holdings + nasdaq_holdings))
                
                # If we couldn't get components, use a default watchlist
                if not scan_symbols:
                    raise ValueError("Could not retrieve index components")
                    
            except Exception as e:
                logger.warning(f"Error getting index components: {e}. Using default watchlist.")
                # Use a default list of liquid stocks across sectors
                scan_symbols = [
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
                    # Others with high volume
                    "F", "T", "GE", "GM", "AAL", "CCL", "DAL", "X", "FCX", "BA"
                ]
            
            # Get current market data for these symbols
            data = {}
            movers = []
            
            # Process in batches to avoid API limits
            batch_size = 20
            for i in range(0, len(scan_symbols), batch_size):
                batch = scan_symbols[i:i+batch_size]
                
                try:
                    # Get data for this batch
                    tickers = yf.Tickers(" ".join(batch))
                    
                    # Process each ticker
                    for symbol in batch:
                        try:
                            # Get history for today and yesterday
                            hist = tickers.tickers[symbol].history(period="2d")
                            
                            if len(hist) >= 2:
                                # Calculate daily change
                                prev_close = hist['Close'].iloc[-2]
                                current = hist['Close'].iloc[-1]
                                volume = hist['Volume'].iloc[-1]
                                
                                # Calculate percentage change
                                change_pct = ((current - prev_close) / prev_close) * 100
                                
                                # Apply filters
                                if current >= min_price and volume >= min_volume:
                                    movers.append({
                                        'symbol': symbol,
                                        'price': current,
                                        'change_pct': change_pct,
                                        'volume': volume,
                                        'prev_close': prev_close
                                    })
                                    
                                    logger.debug(f"Processed {symbol}: ${current:.2f} ({change_pct:.2f}%)")
                                
                        except Exception as e:
                            logger.debug(f"Error processing {symbol}: {e}")
                            continue
                    
                except Exception as e:
                    logger.warning(f"Error processing batch: {e}")
                    continue
            
            # Sort by absolute percentage change
            movers.sort(key=lambda x: abs(x['change_pct']), reverse=True)
            
            # Get top gainers and losers
            gainers = [m for m in movers if m['change_pct'] > 0][:n]
            losers = [m for m in movers if m['change_pct'] < 0][:n]
            
            # Combine top movers
            top_movers = gainers + losers
            
            # Sort by percentage change (highest to lowest)
            top_movers.sort(key=lambda x: x['change_pct'], reverse=True)
            
            logger.info(f"Found {len(top_movers)} top movers")
            return top_movers
            
        except ImportError:
            logger.warning("yfinance not installed. Using simulated top movers.")
            
            # Generate simulated top movers if yfinance isn't available
            watchlist_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]
            
            top_movers = []
            for symbol in watchlist_symbols:
                # Generate random movement between -8% and +8%
                change_pct = np.random.uniform(-8, 8)
                base_price = self.realtime_stream.get_last_price(symbol) or np.random.uniform(50, 500)
                
                top_movers.append({
                    'symbol': symbol,
                    'price': base_price,
                    'change_pct': change_pct,
                    'volume': np.random.randint(500000, 10000000),
                    'prev_close': base_price / (1 + change_pct/100)
                })
            
            # Sort by absolute percentage change
            top_movers.sort(key=lambda x: abs(x['change_pct']), reverse=True)
            
            logger.info(f"Generated {len(top_movers)} simulated top movers")
            return top_movers

    def generate_day_trading_suggestions(self) -> List[Dict[str, Any]]:
        """
        Generate trade suggestions for day trading based on top market movers
        
        Returns:
            List of trade suggestions
        """
        logger.info("Generating day trading suggestions based on top movers...")
        
        try:
            # Get top movers
            top_movers = self.get_top_movers(n=15)  # Get top 15 gainers and losers
            
            if not top_movers:
                logger.warning("No top movers found for day trading")
                return []
            
            # Define strategies for different types of movers
            suggestions = []
            
            # Portfolio value for position sizing
            portfolio_value = self.current_portfolio.get("cash", 0)
            for s, pos in self.current_portfolio.get("positions", {}).items():
                portfolio_value += pos.get("shares", 0) * self.realtime_stream.get_last_price(s)
            
            # Look for patterns in top gainers (momentum plays)
            for mover in [m for m in top_movers if m['change_pct'] > 3.0]:  # Gainers with >3% move
                symbol = mover['symbol']
                price = mover['price']
                change_pct = mover['change_pct']
                
                # Check if we can get intraday data
                try:
                    # Get 5-minute data for pattern analysis
                    intraday_data = self.data_fetcher.fetch_intraday_data(symbol, interval="5m")
                    
                    if intraday_data is not None and not intraday_data.empty:
                        # Basic momentum analysis
                        last_5_candles = intraday_data.tail(5)
                        
                        # Check if stock is in an uptrend (each close higher than previous)
                        uptrend = all(last_5_candles['close'].pct_change().dropna() > 0)
                        
                        # Check for increasing volume
                        volume_trend = last_5_candles['volume'].pct_change().mean() > 0
                        
                        # Calculate RSI for overbought conditions
                        from models.model_trainer import ModelTrainer
                        model_trainer = ModelTrainer()
                        
                        feature_data = model_trainer._engineer_features(
                            intraday_data,
                            ['rsi'],
                            5  # short lookback for intraday
                        )
                        
                        rsi_overbought = feature_data['rsi_14'].iloc[-1] > 70 if 'rsi_14' in feature_data.columns else False
                        
                        # Strategy: Buy momentum breakouts that aren't overbought
                        if uptrend and volume_trend and not rsi_overbought:
                            # Position size: 3% of portfolio per momentum trade
                            position_value = portfolio_value * 0.03
                            quantity = max(1, int(position_value / price))
                            
                            suggestions.append({
                                "symbol": symbol,
                                "action": "BUY",
                                "quantity": quantity,
                                "reason": f"Momentum breakout: +{change_pct:.1f}% with increasing volume",
                                "confidence": min(0.5 + change_pct/20, 0.9),  # Higher change = higher confidence
                                "strategy": "Day Trading - Momentum Breakout",
                                "stop_loss": price * 0.97,  # 3% stop loss
                                "take_profit": price * 1.05  # 5% profit target
                            })
                            
                            logger.info(f"Generated momentum trade for {symbol} +{change_pct:.1f}%")
                    
                except Exception as e:
                    logger.warning(f"Error analyzing intraday data for {symbol}: {e}")
                    
                    # Fallback: Use just the daily move if intraday analysis fails
                    if change_pct > 5.0 and change_pct < 20.0:  # Significant move but not excessive
                        # Position size: 2% of portfolio
                        position_value = portfolio_value * 0.02
                        quantity = max(1, int(position_value / price))
                        
                        suggestions.append({
                            "symbol": symbol,
                            "action": "BUY",
                            "quantity": quantity,
                            "reason": f"Strong momentum: +{change_pct:.1f}% daily move",
                            "confidence": 0.65,
                            "strategy": "Day Trading - Daily Momentum",
                            "stop_loss": price * 0.96,  # 4% stop loss
                            "take_profit": price * 1.04  # 4% profit target
                        })
            
            # Look for reversal setups in top losers
            for mover in [m for m in top_movers if m['change_pct'] < -4.0]:  # Losers with >4% drop
                symbol = mover['symbol']
                price = mover['price']
                change_pct = mover['change_pct']
                
                # A simple reversal strategy based on oversold conditions
                try:
                    # Get daily data for pattern analysis
                    daily_data = self.data_fetcher.fetch_daily_data(symbol, days=20)
                    
                    if daily_data is not None and not daily_data.empty:
                        # Create features for analysis
                        from models.model_trainer import ModelTrainer
                        model_trainer = ModelTrainer()
                        
                        feature_data = model_trainer._engineer_features(
                            daily_data,
                            ['rsi', 'bollinger'],
                            10
                        )
                        
                        # Check for oversold conditions
                        rsi_oversold = feature_data['rsi_14'].iloc[-1] < 30 if 'rsi_14' in feature_data.columns else False
                        
                        # Check if price is near lower Bollinger Band
                        near_lower_band = False
                        if 'bollinger_lower' in feature_data.columns:
                            lower_band = feature_data['bollinger_lower'].iloc[-1]
                            near_lower_band = (price / lower_band - 1) < 0.02  # Within 2% of lower band
                        
                        # Only trade reversals if stock is oversold or near support level
                        if rsi_oversold or near_lower_band:
                            # Smaller position for reversal trades (more risky)
                            position_value = portfolio_value * 0.02
                            quantity = max(1, int(position_value / price))
                            
                            suggestions.append({
                                "symbol": symbol,
                                "action": "BUY",
                                "quantity": quantity,
                                "reason": f"Potential reversal: {change_pct:.1f}% drop, {'oversold' if rsi_oversold else 'near support'}",
                                "confidence": 0.6,  # Lower confidence for reversal trades
                                "strategy": "Day Trading - Oversold Reversal",
                                "stop_loss": price * 0.95,  # 5% stop loss
                                "take_profit": price * 1.05  # 5% profit target
                            })
                            
                            logger.info(f"Generated reversal trade for {symbol} {change_pct:.1f}%")
                
                except Exception as e:
                    logger.warning(f"Error analyzing data for reversal on {symbol}: {e}")
            
            # Validate against risk management rules
            valid_suggestions = []
            for suggestion in suggestions:
                # Use risk manager to validate the trade
                is_valid, reason = self.risk_manager.validate_trade(
                    trade_symbol=suggestion["symbol"],
                    trade_size=suggestion.get("quantity", 0) * suggestion.get("price", 0),
                    trade_direction=suggestion["action"],
                    current_portfolio=self.current_portfolio.get("positions", {}),
                    historical_data={},
                    sector_data={}
                )
                
                if is_valid:
                    valid_suggestions.append(suggestion)
                    logger.info(f"Generated valid day trading suggestion: {suggestion['action']} {suggestion['quantity']} {suggestion['symbol']}")
                else:
                    logger.warning(f"Day trading suggestion rejected by risk management: {reason}")
            
            # Store and return suggestions
            self.trade_suggestions = valid_suggestions
            return valid_suggestions
            
        except Exception as e:
            logger.error(f"Error generating day trading suggestions: {e}")
            return []

    def setup_alert_system(self, symbols=None) -> None:
        """
        Set up the alert system for intraday monitoring
        
        Args:
            symbols: List of symbols to monitor (None = use top movers)
        """
        logger.info("Setting up alert system...")
        
        try:
            # Import alert system
            from trading.alert_system import AlertSystem
            
            # Initialize alert system
            self.alert_system = AlertSystem(
                data_fetcher=self.data_fetcher,
                realtime_stream=self.realtime_stream
            )
            
            # Define callback for alerts
            def alert_callback(alert):
                symbol = alert.get('symbol')
                alert_type = alert.get('alert_type')
                logger.info(f"Alert received: {alert_type} for {symbol}")
                
                # Add to trade suggestions if it's a significant alert
                if alert_type in ['price_breakout_up', 'rsi_oversold', 'ma_cross_bullish']:
                    # Create a trade suggestion from the alert
                    price = alert.get('data', {}).get('price', 0)
                    
                    # Calculate position size (1% of portfolio)
                    portfolio_value = self.current_portfolio.get("cash", 0)
                    for s, pos in self.current_portfolio.get("positions", {}).items():
                        portfolio_value += pos.get("shares", 0) * self.realtime_stream.get_last_price(s)
                    
                    position_value = portfolio_value * 0.01
                    quantity = max(1, int(position_value / price)) if price > 0 else 0
                    
                    suggestion = {
                        "symbol": symbol,
                        "action": "BUY",
                        "quantity": quantity,
                        "reason": f"Alert: {alert_type}",
                        "confidence": 0.7,
                        "strategy": "Intraday Alert System",
                        "alert_data": alert.get('data', {})
                    }
                    
                    # Add to suggestions if not already there
                    if not any(s["symbol"] == symbol for s in self.trade_suggestions):
                        self.trade_suggestions.append(suggestion)
            
            # Add the callback
            self.alert_system.add_alert_callback(alert_callback)
            
            # If symbols not provided, use top movers
            if symbols is None:
                top_movers = self.get_top_movers(n=10)
                symbols = [m['symbol'] for m in top_movers]
            
            # Start monitoring
            self.alert_system.start_monitoring(symbols, interval='5m')
            
            logger.info(f"Alert system set up and monitoring {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error setting up alert system: {e}")

    def get_pending_alerts(self) -> List[Dict[str, Any]]:
        """
        Get pending alerts from the alert system
        
        Returns:
            List of pending alerts
        """
        if hasattr(self, 'alert_system'):
            return self.alert_system.get_pending_alerts()
        else:
            return []

    def find_alert_based_trades(self) -> List[Dict[str, Any]]:
        """
        Find potential trades based on recent alerts
        
        Returns:
            List of potential trades
        """
        if hasattr(self, 'alert_system'):
            return self.alert_system.find_potential_trades()
        else:
            return []


# Add placeholder for TradeJournaler
class TradeJournaler:
    def get_trades_by_date(self, start_date, end_date):
        # Return empty list for now
        return []
    
    def record_trade(self, trade):
        # Do nothing for now
        pass


# Add properties to SmartTradeAdvisor class to make app.py work
SmartTradeAdvisor.trade_journaler = property(lambda self: TradeJournaler())
SmartTradeAdvisor.risk_manager = property(lambda self: RiskManager())


def run_cli():
    """Command line interface to run the trading advisor"""
    advisor = SmartTradeAdvisor()
    advisor.start()
    
    try:
        while True:
            print("\nSmart Trade Advisor Menu:")
            print("1. Run analysis and get trade suggestions")
            print("2. Find day trading opportunities (top movers)")
            print("3. Set up real-time alert monitoring")
            print("4. Scanner presets (gap plays, breakouts, etc.)")
            print("5. Analyze sector strength and rotation")
            print("6. Get news and catalysts for stocks")
            print("7. Check pending alerts & alert-based trades")
            print("8. Execute suggested trades")
            print("9. View portfolio")
            print("10. Train ML models")
            print("11. Exit")
            
            choice = input("Enter your choice (1-11): ")
            
            if choice == "1":
                use_ml = input("Use ML-based predictions? (y/n): ").lower() == 'y'
                
                if use_ml and not advisor.check_ml_dependencies():
                    print("ML dependencies not installed. Please run install-ml-packages.bat first.")
                    continue
                
                suggestions = advisor.generate_trade_suggestions()
                
                if not suggestions:
                    print("No trade suggestions generated for current market conditions.")
                else:
                    print("\nTrade Suggestions:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"{i}. {suggestion['action']} {suggestion['quantity']} {suggestion['symbol']} - {suggestion['reason']}")
                        print(f"   Strategy: {suggestion['strategy']}, Confidence: {suggestion['confidence']:.2%}")
            
            elif choice == "2":
                print("\nScanning for top market movers for day trading...")
                suggestions = advisor.generate_day_trading_suggestions()
                
                if not suggestions:
                    print("No day trading opportunities identified in current market conditions.")
                else:
                    print("\nDay Trading Opportunities:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"{i}. {suggestion['action']} {suggestion['quantity']} {suggestion['symbol']} - {suggestion['reason']}")
                        print(f"   Strategy: {suggestion['strategy']}, Confidence: {suggestion['confidence']:.2%}")
                        print(f"   Risk Management: Stop Loss: ${suggestion.get('stop_loss', 0):.2f}, Take Profit: ${suggestion.get('take_profit', 0):.2f}")
            
            elif choice == "3":
                print("\nSetting up real-time alert monitoring system...")
                print("1. Monitor top movers automatically")
                print("2. Specify symbols to monitor")
                print("3. Customize alert settings")
                print("4. Back to main menu")
                
                alert_choice = input("Enter your choice (1-4): ")
                
                if alert_choice == "1":
                    print("Setting up alert monitoring for top market movers...")
                    advisor.setup_alert_system()
                    print("Alert system is now monitoring top movers. You'll be notified of patterns as they form.")
                
                elif alert_choice == "2":
                    symbols_input = input("Enter symbols to monitor (comma-separated, e.g., AAPL,MSFT,TSLA): ")
                    symbols = [s.strip().upper() for s in symbols_input.split(',')]
                    if symbols:
                        advisor.setup_alert_system(symbols)
                        print(f"Alert system is now monitoring {len(symbols)} symbols: {', '.join(symbols)}")
                    else:
                        print("No symbols entered.")
                
                elif alert_choice == "3":
                    print("\nCustomize Alert Settings:")
                    print("1. Volume surge threshold")
                    print("2. RSI overbought/oversold levels")
                    print("3. Moving average periods")
                    print("4. Back to alert menu")
                    
                    setting_choice = input("Enter setting to change (1-4): ")
                    
                    if setting_choice == "1":
                        threshold = input("Enter new volume surge threshold (default 2.0): ")
                        try:
                            threshold = float(threshold)
                            if hasattr(advisor, 'alert_system'):
                                advisor.alert_system.update_alert_settings({
                                    'volume_surge': {'threshold': threshold}
                                })
                                print(f"Volume surge threshold updated to {threshold}")
                            else:
                                print("Alert system not initialized. Set up monitoring first.")
                        except ValueError:
                            print("Invalid value. Please enter a number.")
                    
                    elif setting_choice == "2":
                        overbought = input("Enter RSI overbought level (default 70): ")
                        oversold = input("Enter RSI oversold level (default 30): ")
                        try:
                            overbought = float(overbought)
                            oversold = float(oversold)
                            if hasattr(advisor, 'alert_system'):
                                advisor.alert_system.update_alert_settings({
                                    'rsi_extreme': {
                                        'overbought': overbought,
                                        'oversold': oversold
                                    }
                                })
                                print(f"RSI levels updated: Overbought={overbought}, Oversold={oversold}")
                            else:
                                print("Alert system not initialized. Set up monitoring first.")
                        except ValueError:
                            print("Invalid value. Please enter numbers.")
                            
                    elif setting_choice == "3":
                        fast_period = input("Enter fast MA period (default 5): ")
                        slow_period = input("Enter slow MA period (default 20): ")
                        try:
                            fast_period = int(fast_period)
                            slow_period = int(slow_period)
                            if hasattr(advisor, 'alert_system'):
                                advisor.alert_system.update_alert_settings({
                                    'moving_average_cross': {
                                        'fast_period': fast_period,
                                        'slow_period': slow_period
                                    }
                                })
                                print(f"Moving average periods updated: Fast={fast_period}, Slow={slow_period}")
                            else:
                                print("Alert system not initialized. Set up monitoring first.")
                        except ValueError:
                            print("Invalid value. Please enter integers.")
            
            elif choice == "4":
                print("\nScanner Presets:")
                print("1. Gap Plays - Stocks with significant overnight gaps")
                print("2. Breakouts - Stocks breaking out of consolidation patterns")
                print("3. Oversold Reversals - Potential bounce candidates")
                print("4. Earnings Movers - Stocks with significant post-earnings moves")
                print("5. Back to main menu")
                
                scanner_choice = input("Select scanner preset (1-5): ")
                
                try:
                    from trading.scanner_presets import ScannerPresets
                    scanner = ScannerPresets(data_fetcher=advisor.data_fetcher, realtime_stream=advisor.realtime_stream)
                    
                    if scanner_choice == "1":
                        print("\nScanning for gap plays...")
                        gap_plays = scanner.scan_gap_plays()
                        
                        if gap_plays:
                            print(f"\nFound {len(gap_plays)} gap plays:")
                            print("\nGap Ups:")
                            for play in [p for p in gap_plays if p['gap_direction'] == 'up'][:5]:
                                print(f"  {play['symbol']}: +{play['gap_percent']:.2f}%, Price: ${play['price']:.2f}, Volume: {play['volume']:,}")
                            
                            print("\nGap Downs:")
                            for play in [p for p in gap_plays if p['gap_direction'] == 'down'][:5]:
                                print(f"  {play['symbol']}: {play['gap_percent']:.2f}%, Price: ${play['price']:.2f}, Volume: {play['volume']:,}")
                        else:
                            print("No gap plays found matching criteria.")
                    
                    elif scanner_choice == "2":
                        print("\nScanning for breakouts...")
                        breakouts = scanner.scan_breakouts()
                        
                        if breakouts:
                            print(f"\nFound {len(breakouts)} breakout candidates:")
                            for b in breakouts[:10]:
                                print(f"  {b['symbol']}: Breaking out +{b['breakout_percent']:.2f}% above ${b['resistance_level']:.2f}")
                                print(f"    Current: ${b['price']:.2f}, Volume surge: {b['volume_ratio']:.1f}x")
                        else:
                            print("No breakout candidates found matching criteria.")
                    
                    elif scanner_choice == "3":
                        print("\nScanning for oversold reversal candidates...")
                        reversals = scanner.scan_oversold_reversals()
                        
                        if reversals:
                            print(f"\nFound {len(reversals)} oversold reversal candidates:")
                            for r in reversals[:10]:
                                print(f"  {r['symbol']}: RSI {r['rsi']:.1f}, Price: ${r['price']:.2f}")
                                print(f"    {'Near support level, ' if r['near_support'] else ''}Daily change: {r['price_change_pct']:.2f}%")
                        else:
                            print("No oversold reversal candidates found matching criteria.")
                    
                    elif scanner_choice == "4":
                        print("\nScanning for earnings movers...")
                        earnings_movers = scanner.scan_earnings_movers()
                        
                        if earnings_movers:
                            print(f"\nFound {len(earnings_movers)} stocks with significant earnings moves:")
                            
                            print("\nUp Moves:")
                            for mover in [m for m in earnings_movers if m['move_direction'] == 'up'][:5]:
                                print(f"  {mover['symbol']}: +{mover['move_percent']:.2f}%, Earnings: {mover['earnings_date']}")
                                print(f"    Current price: ${mover['price']:.2f}, Days since earnings: {mover['days_since_earnings']}")
                            
                            print("\nDown Moves:")
                            for mover in [m for m in earnings_movers if m['move_direction'] == 'down'][:5]:
                                print(f"  {mover['symbol']}: {mover['move_percent']:.2f}%, Earnings: {mover['earnings_date']}")
                                print(f"    Current price: ${mover['price']:.2f}, Days since earnings: {mover['days_since_earnings']}")
                        else:
                            print("No significant earnings movers found.")
                
                except ImportError:
                    print("Scanner presets not available. Please run install-day-trading.bat to install required packages.")
            
            elif choice == "5":
                print("\nSector Strength Analysis:")
                
                try:
                    from trading.sector_analysis import SectorAnalysis
                    sector_analyzer = SectorAnalysis(data_fetcher=advisor.data_fetcher)
                    
                    print("Analyzing sector performance...")
                    sector_data = sector_analyzer.analyze_sector_performance()
                    
                    if sector_data and 'sectors' in sector_data:
                        print(f"\nMarket Phase: {sector_data.get('market_phase', 'Unknown')}")
                        
                        print("\nLeading Sectors:")
                        for sector in sector_data.get('leading_sectors', []):
                            print(f"  {sector}")
                        
                        print("\nLagging Sectors:")
                        for sector in sector_data.get('lagging_sectors', []):
                            print(f"  {sector}")
                        
                        print("\nSector Performance (1 Week):")
                        for sector in sector_data.get('sectors', [])[:5]:
                            print(f"  {sector['sector']}: {sector['returns_1w']:.2f}% ({sector['trend']})")
                            print(f"    Relative to S&P 500: {sector['relative_strength_1w']:.2f}%")
                        
                        # Find strong sectors for trading
                        strong_sectors = sector_analyzer.find_strong_sectors()
                        
                        if strong_sectors:
                            print("\nStrong Sectors for Trading Opportunities:")
                            for sector in strong_sectors:
                                print(f"  {sector['sector']} (ETF: {sector['etf']}): {sector['relative_strength']:.2f}% rel. strength")
                                print(f"    Key stocks: {', '.join(sector['key_stocks'])}")
                        
                        print("\nFull sector analysis saved to results/sectors directory.")
                    else:
                        print("No sector data available.")
                
                except ImportError:
                    print("Sector analysis not available. Please run install-day-trading.bat to install required packages.")
            
            elif choice == "6":
                print("\nNews & Catalysts Integration:")
                print("1. Get news for a specific stock")
                print("2. Get general market news")
                print("3. Find stocks with catalysts")
                print("4. Back to main menu")
                
                news_choice = input("Enter your choice (1-4): ")
                
                try:
                    from trading.news_integration import NewsIntegration
                    news_system = NewsIntegration()
                    
                    if news_choice == "1":
                        symbol = input("Enter stock symbol: ").upper()
                        if symbol:
                            print(f"\nGetting news for {symbol}...")
                            news = news_system.get_stock_news(symbol)
                            
                            if news:
                                print(f"\nLatest news for {symbol}:")
                                for i, item in enumerate(news, 1):
                                    print(f"{i}. {item['title']}")
                                    print(f"   Source: {item['source']}, Date: {item['date']}")
                                    if 'sentiment' in item:
                                        sentiment = item['sentiment']
                                        sentiment_str = "Positive" if sentiment > 0.2 else "Negative" if sentiment < -0.2 else "Neutral"
                                        print(f"   Sentiment: {sentiment_str} ({sentiment:.2f})")
                                    if 'summary' in item and item['summary']:
                                        print(f"   Summary: {item['summary'][:100]}...")
                                    print()
                            else:
                                print(f"No recent news found for {symbol}")
                    
                    elif news_choice == "2":
                        print("\nGetting market news...")
                        market_news = news_system.get_market_news()
                        
                        if market_news:
                            print("\nLatest Market News:")
                            for i, item in enumerate(market_news, 1):
                                print(f"{i}. {item['title']}")
                                print(f"   Source: {item['source']}, Date: {item['date']}")
                                if 'sentiment' in item:
                                    sentiment = item['sentiment']
                                    sentiment_str = "Positive" if sentiment > 0.2 else "Negative" if sentiment < -0.2 else "Neutral"
                                    print(f"   Sentiment: {sentiment_str} ({sentiment:.2f})")
                                print()
                        else:
                            print("No market news available")
                    
                    elif news_choice == "3":
                        print("\nScanning for stocks with significant catalysts...")
                        catalyst_stocks = news_system.find_catalyst_stocks()
                        
                        if catalyst_stocks:
                            print(f"\nFound {len(catalyst_stocks)} stocks with recent catalysts:")
                            for stock in catalyst_stocks:
                                print(f"\n{stock['symbol']} - {stock['catalyst_count']} catalysts")
                                for i, catalyst in enumerate(stock['catalysts'], 1):
                                    print(f"  {i}. {catalyst['title']}")
                                    print(f"     Source: {catalyst.get('source', 'Unknown')}, Date: {catalyst.get('date', 'Unknown')}")
                                    if 'sentiment' in catalyst:
                                        sentiment = catalyst['sentiment']
                                        sentiment_str = "Positive" if sentiment > 0.2 else "Negative" if sentiment < -0.2 else "Neutral"
                                        print(f"     Sentiment: {sentiment_str} ({sentiment:.2f})")
                        else:
                            print("No stocks with significant catalysts found")
                
                except ImportError:
                    print("News integration not available. Please run install-day-trading.bat to install required packages.")
            
            elif choice == "7":
                print("\nAlert System Status:")
                
                if hasattr(advisor, 'alert_system'):
                    # Check for pending alerts
                    pending_alerts = advisor.get_pending_alerts()
                    
                    if pending_alerts:
                        print(f"\nYou have {len(pending_alerts)} new alerts:")
                        for alert in pending_alerts:
                            symbol = alert.get('symbol')
                            alert_type = alert.get('alert_type')
                            price = alert.get('data', {}).get('price', 0)
                            print(f"  {symbol}: {alert_type.replace('_', ' ').title()} at ${price:.2f}")
                    else:
                        print("No new alerts at this time")
                    
                    # Get potential trades based on alerts
                    alert_trades = advisor.find_alert_based_trades()
                    
                    if alert_trades:
                        print(f"\nPotential trades based on alerts ({len(alert_trades)}):")
                        for i, trade in enumerate(alert_trades, 1):
                            print(f"{i}. {trade['action']} {trade['symbol']} - {trade['setup']}")
                            print(f"   Price: ${trade['price']:.2f}, Confidence: {trade['confidence']:.2%}")
                            if 'stop_loss' in trade:
                                print(f"   Stop Loss: ${trade['stop_loss']:.2f}")
                        
                        # Ask if user wants to add these to suggestions
                        add_to_suggestions = input("\nAdd these alert-based trades to suggestions list? (y/n): ")
                        if add_to_suggestions.lower() == 'y':
                            for trade in alert_trades:
                                # Calculate quantity based on portfolio value and position sizing
                                portfolio_value = advisor.current_portfolio.get("cash", 0)
                                for s, pos in advisor.current_portfolio.get("positions", {}).items():
                                    portfolio_value += pos.get("shares", 0) * advisor.realtime_stream.get_last_price(s)
                                
                                # Position size: 2% of portfolio per trade
                                position_value = portfolio_value * 0.02
                                price = trade['price']
                                quantity = max(1, int(position_value / price)) if price > 0 else 0
                                
                                suggestion = {
                                    "symbol": trade['symbol'],
                                    "action": trade['action'],
                                    "quantity": quantity,
                                    "reason": f"Alert-based {trade['setup']}",
                                    "confidence": trade['confidence'],
                                    "strategy": "Alert System",
                                    "stop_loss": trade.get('stop_loss')
                                }
                                
                                # Add to suggestions if not already there
                                if not any(s["symbol"] == trade["symbol"] for s in advisor.trade_suggestions):
                                    advisor.trade_suggestions.append(suggestion)
                            
                            print(f"Added {len(alert_trades)} alert-based trades to suggestions")
                    else:
                        print("No potential trades based on recent alerts")
                    
                    # View alert history
                    view_history = input("\nView alert history? (y/n): ")
                    if view_history.lower() == 'y':
                        alerts = advisor.alert_system.get_alert_history(days=1)
                        if alerts:
                            print(f"\nAlert history (last 24 hours, {len(alerts)} alerts):")
                            for i, alert in enumerate(alerts[:10], 1):  # Show top 10
                                alert_time = alert.get('time')
                                time_str = alert_time.strftime('%H:%M:%S') if isinstance(alert_time, datetime) else str(alert_time)
                                print(f"{i}. {time_str} - {alert.get('symbol')}: {alert.get('alert_type').replace('_', ' ').title()}")
                        else:
                            print("No alerts in history")
                else:
                    print("Alert system not set up. Please use option 3 to set up real-time alerts.")
            
            elif choice == "8":
                if not advisor.trade_suggestions:
                    print("No trade suggestions available. Run analysis, scanner, or check alerts first.")
                    continue
                    
                print("\nExecute Trade Suggestions:")
                for i, suggestion in enumerate(advisor.trade_suggestions, 1):
                    print(f"{i}. {suggestion['action']} {suggestion['quantity']} {suggestion['symbol']} - {suggestion['reason']}")
                    print(f"   Strategy: {suggestion['strategy']}, Confidence: {suggestion['confidence']:.2%}")
                    if 'stop_loss' in suggestion:
                        print(f"   Stop Loss: ${suggestion['stop_loss']:.2f}")
                
                try:
                    exec_id = int(input("\nEnter suggestion number to execute (0 to cancel): "))
                    if exec_id > 0 and exec_id <= len(advisor.trade_suggestions):
                        suggestion = advisor.trade_suggestions[exec_id-1]
                        confirm = input(f"Confirm {suggestion['action']} {suggestion['quantity']} {suggestion['symbol']}? (y/n): ")
                        if confirm.lower() == 'y':
                            result = advisor.execute_trade(suggestion)
                            print(f"Trade executed: {result['status']}")
                            print(f"Trade ID: {result['trade_id']}")
                            print(f"Executed Price: ${result['executed_price']:.2f}")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            elif choice == "9":
                print("\nCurrent Portfolio:")
                portfolio = advisor.current_portfolio
                print(f"Cash: ${portfolio.get('cash', 0):,.2f}")
                
                # Calculate total portfolio value
                total_value = portfolio.get('cash', 0)
                
                print("\nPositions:")
                positions = portfolio.get('positions', {})
                if positions:
                    for symbol, pos in positions.items():
                        price = advisor.realtime_stream.get_last_price(symbol) or pos.get('average_price', 0)
                        value = pos.get('shares', 0) * price
                        total_value += value
                        
                        # Calculate profit/loss
                        cost_basis = pos.get('shares', 0) * pos.get('average_price', 0)
                        pnl = value - cost_basis
                        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
                        
                        print(f"  {symbol}: {pos.get('shares', 0)} shares @ ${price:.2f} = ${value:,.2f}")
                        print(f"    P&L: ${pnl:,.2f} ({pnl_pct:.2f}%)")
                else:
                    print("  No positions")
                
                print(f"\nTotal Portfolio Value: ${total_value:,.2f}")
            
            elif choice == "10":
                print("\nTrain Machine Learning Models:")
                if not advisor.check_ml_dependencies():
                    print("ML dependencies not installed. Please run install-ml-packages.bat first.")
                    continue
                
                print("Available model types: xgboost, random_forest, gradient_boosting, ridge")
                model_type = input("Enter model type (default: xgboost): ") or "xgboost"
                
                # Give option to train on custom symbols or top movers
                use_top_movers = input("Train on top market movers? (y/n, default=n): ").lower() == 'y'
                
                if use_top_movers:
                    print("Getting top market movers...")
                    top_movers = advisor.get_top_movers(n=15)
                    symbols = [m['symbol'] for m in top_movers]
                    print(f"Training on {len(symbols)} top movers: {', '.join(symbols)}")
                else:
                    custom_symbols = input("Enter symbols to train on (comma-separated, blank for default): ")
                    if custom_symbols:
                        symbols = [s.strip().upper() for s in custom_symbols.split(',')]
                    else:
                        symbols = None
                
                print("Training models (this may take a few minutes)...")
                result = advisor.train_models(symbols=symbols, model_type=model_type)
                
                if result["status"] == "success":
                    print(f"Models trained successfully for {len(result['trained_symbols'])} symbols!")
                    print("\nPerformance Summary:")
                    for symbol, perf in result["performance"].items():
                        print(f"  {symbol}:")
                        print(f"    Direction Accuracy: {perf['direction_accuracy']:.2%}")
                        print(f"    Profit Factor: {perf['profit_factor']:.2f}")
                else:
                    print(f"Error training models: {result.get('error', 'Unknown error')}")
            
            elif choice == "11":
                break
            
            else:
                print("Invalid choice. Please enter a number between 1 and 11.")
    
    except KeyboardInterrupt:
        print("\nShutting down Smart Trade Advisor...")
    finally:
        # Stop any running systems
        if hasattr(advisor, 'alert_system'):
            advisor.alert_system.stop_monitoring_alerts()
        advisor.stop()


if __name__ == "__main__":
    run_cli() 