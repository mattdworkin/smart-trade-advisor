from backtesting.backtest import Backtest, DataHandler
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# For time series models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Trains machine learning models for market prediction.
    """
    
    def __init__(self, data_dir: str = 'data/processed'):
        """
        Initialize the model trainer.
        
        Args:
            data_dir: Directory to store/load processed data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Directory to save trained models
        self.models_dir = 'models/trained'
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Available model types
        self.model_types = {
            'random_forest': self._create_random_forest,
            'gradient_boosting': self._create_gradient_boosting,
            'xgboost': self._create_xgboost,
            'ridge': self._create_ridge
        }
    
    def train_model(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train a model based on the provided configuration.
        
        Args:
            config: Dictionary containing training configuration parameters
                    
        Returns:
            Dictionary containing the trained model and metadata
        """
        if config is None:
            config = {
                'model_type': 'xgboost',
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
                'prediction_target': 'next_day_return',
                'prediction_horizon': 1,  # days
                'features': ['rsi', 'macd', 'bollinger', 'volume'],
                'lookback_period': 30,  # days
                'test_size': 0.2,
                'cv_folds': 3,
                'run_backtest': True
            }
        
        logger.info(f"Training model with configuration: {config}")
        
        # Get historical data
        symbols = config.get('symbols', ['AAPL'])
        raw_data = self._get_historical_data(symbols)
        
        # Process data for all symbols
        processed_data = {}
        models = {}
        performance = {}
        
        for symbol in symbols:
            if symbol in raw_data and not raw_data[symbol].empty:
                # Extract and prepare data for this symbol
                symbol_data = raw_data[symbol].copy()
                
                # Engineer features
                feature_data = self._engineer_features(
                    symbol_data, 
                    config.get('features', []), 
                    config.get('lookback_period', 30)
                )
                
                # Create target variable
                horizon = config.get('prediction_horizon', 1)
                target_data = self._create_target(
                    feature_data, 
                    config.get('prediction_target', 'next_day_return'),
                    horizon
                )
                
                # Drop NaN values created by feature engineering
                target_data = target_data.dropna()
                
                # Save processed data
                processed_data[symbol] = target_data
                
                # Train the model
                logger.info(f"Training model for {symbol}...")
                model, model_performance = self._train_symbol_model(
                    target_data, 
                    config.get('model_type', 'xgboost'),
                    config.get('test_size', 0.2),
                    config.get('cv_folds', 3)
                )
                
                models[symbol] = model
                performance[symbol] = model_performance
                
                # Save the model
                model_path = os.path.join(self.models_dir, f"{symbol}_{config.get('model_type')}.joblib")
                joblib.dump(model, model_path)
                logger.info(f"Model saved to {model_path}")
                
                # Generate performance visualization
                self._plot_performance(symbol, target_data, model, model_performance)
            else:
                logger.warning(f"No data available for {symbol}, skipping model training")
        
        # Prepare the complete model package
        trained_model = {
            'type': config.get('model_type'),
            'models': models,
            'config': config,
            'performance': performance,
            'trained_date': datetime.now().isoformat(),
            'symbols': symbols
        }
        
        logger.info("Model training complete!")
        
        # Run backtest if configured
        if config.get('run_backtest', False):
            try:
                data_handler = DataHandler(start_date='2022-01-01', end_date='2023-12-31')
                backtest = Backtest(trained_model)
                metrics = backtest.run_walk_forward_test(data_handler)
                logger.info(f"\nBacktest Metrics:\n{metrics}")
                
                # Save metrics to CSV
                metrics_dir = 'results'
                os.makedirs(metrics_dir, exist_ok=True)
                metrics_path = os.path.join(metrics_dir, f'backtest_metrics_{datetime.now().strftime("%Y%m%d")}.csv')
                metrics.to_csv(metrics_path)
                logger.info(f"Metrics saved to {metrics_path}")
                
                # Add backtest results to the model package
                trained_model['backtest_metrics'] = metrics
            except Exception as e:
                logger.error(f"Error during backtesting: {e}")
        
        return trained_model
    
    def _get_historical_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for the given symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary of symbol -> price DataFrame
        """
        logger.info(f"Fetching historical data for {len(symbols)} symbols")
        
        try:
            # Try to import your historical data fetcher
            from data.historical_data_fetcher import HistoricalDataFetcher
            data_fetcher = HistoricalDataFetcher()
            
            data = {}
            for symbol in symbols:
                try:
                    # Fetch 5 years of daily data
                    symbol_data = data_fetcher.fetch_daily_data(
                        symbol, 
                        start_date=(datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d'),
                        end_date=datetime.now().strftime('%Y-%m-%d')
                    )
                    
                    if symbol_data is not None and not symbol_data.empty:
                        data[symbol] = symbol_data
                        logger.info(f"Fetched {len(symbol_data)} data points for {symbol}")
                    else:
                        logger.warning(f"No data returned for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
            
            return data
            
        except ImportError:
            logger.warning("Could not import HistoricalDataFetcher, using Yahoo Finance instead")
            
            # Fallback to yfinance
            try:
                import yfinance as yf
                
                data = {}
                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        symbol_data = ticker.history(period="5y")
                        
                        # Rename columns to match expected format
                        symbol_data.columns = [col.lower() for col in symbol_data.columns]
                        
                        if not symbol_data.empty:
                            data[symbol] = symbol_data
                            logger.info(f"Fetched {len(symbol_data)} data points for {symbol}")
                        else:
                            logger.warning(f"No data returned for {symbol}")
                    except Exception as e:
                        logger.error(f"Error fetching data for {symbol}: {e}")
                
                return data
                
            except ImportError:
                logger.error("Could not import yfinance. Using synthetic data instead.")
                
                # Generate synthetic data as last resort
                data = {}
                for symbol in symbols:
                    dates = pd.date_range(end=datetime.now(), periods=1000)
                    
                    # Simulate price movement
                    base_price = np.random.uniform(50, 500)
                    trend = np.random.uniform(-0.0001, 0.0003)
                    volatility = np.random.uniform(0.005, 0.02)
                    
                    # Generate log returns
                    log_returns = np.random.normal(trend, volatility, len(dates))
                    # Convert to prices
                    prices = base_price * np.exp(np.cumsum(log_returns))
                    
                    # Create OHLC data
                    df = pd.DataFrame({
                        'open': prices * np.random.uniform(0.99, 1.01, len(dates)),
                        'high': prices * np.random.uniform(1.01, 1.03, len(dates)),
                        'low': prices * np.random.uniform(0.97, 0.99, len(dates)),
                        'close': prices,
                        'volume': np.random.normal(1e6, 2e5, len(dates)).astype(int)
                    }, index=dates)
                    
                    data[symbol] = df
                    logger.info(f"Generated synthetic data for {symbol}")
                
                return data
    
    def _engineer_features(self, 
                          data: pd.DataFrame, 
                          feature_types: List[str],
                          lookback_period: int) -> pd.DataFrame:
        """
        Engineer features for model training.
        
        Args:
            data: DataFrame with price data
            feature_types: List of feature types to generate
            lookback_period: Period for lookback features
            
        Returns:
            DataFrame with original data and engineered features
        """
        df = data.copy()
        
        # Make sure we have the necessary price columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Missing some required columns. Available: {df.columns}")
            # Fill in any missing columns with close
            for col in required_columns:
                if col not in df.columns and 'close' in df.columns:
                    df[col] = df['close']
                elif col not in df.columns:
                    logger.error(f"Cannot create {col} column, no close price available")
                    return df
        
        # Create features based on types
        for feature_type in feature_types:
            if feature_type == 'rsi':
                # RSI - Relative Strength Index
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                df['rsi_14'] = 100 - (100 / (1 + rs))
                
            elif feature_type == 'macd':
                # MACD - Moving Average Convergence Divergence
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema12 - ema26
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
                
            elif feature_type == 'bollinger':
                # Bollinger Bands
                df['ma20'] = df['close'].rolling(window=20).mean()
                df['stddev'] = df['close'].rolling(window=20).std()
                df['bollinger_upper'] = df['ma20'] + 2 * df['stddev']
                df['bollinger_lower'] = df['ma20'] - 2 * df['stddev']
                df['bollinger_pct'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
                
            elif feature_type == 'volume':
                # Volume indicators
                df['volume_ma20'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma20']
                df['on_balance_volume'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            elif feature_type == 'momentum':
                # Momentum indicators
                df['returns_1d'] = df['close'].pct_change(1)
                df['returns_5d'] = df['close'].pct_change(5)
                df['returns_10d'] = df['close'].pct_change(10)
                df['returns_20d'] = df['close'].pct_change(20)
            
            elif feature_type == 'trend':
                # Trend indicators
                df['ma50'] = df['close'].rolling(window=50).mean()
                df['ma200'] = df['close'].rolling(window=200).mean()
                df['ma_ratio_50_200'] = df['ma50'] / df['ma200']
                
            elif feature_type == 'volatility':
                # Volatility indicators
                df['volatility_20d'] = df['returns_1d'].rolling(window=20).std() * np.sqrt(252)  # Annualize
                
            elif feature_type == 'seasonality':
                # Seasonality features
                df['day_of_week'] = df.index.dayofweek
                df['month'] = df.index.month
                df['quarter'] = df.index.quarter
        
        # Create lagged features
        price_cols = ['close', 'high', 'low', 'open']
        for col in price_cols:
            if col in df.columns:
                for lag in range(1, lookback_period + 1):
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Create rolling statistics for close price
        windows = [5, 10, 20, 50]
        for window in windows:
            if 'close' in df.columns:
                df[f'close_ma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
                
                # Price relative to moving average
                df[f'close_ma_ratio_{window}'] = df['close'] / df[f'close_ma_{window}']
        
        return df
    
    def _create_target(self, 
                      data: pd.DataFrame, 
                      target_type: str,
                      horizon: int) -> pd.DataFrame:
        """
        Create the target variable for supervised learning.
        
        Args:
            data: DataFrame with price and features
            target_type: Type of prediction target
            horizon: Prediction horizon in days
            
        Returns:
            DataFrame with added target variable
        """
        df = data.copy()
        
        if target_type == 'next_day_return':
            # Predict the next day's return
            df['target'] = df['close'].pct_change(-1)  # Next day's return
            
        elif target_type == 'next_day_direction':
            # Predict the direction (up/down)
            df['target'] = np.sign(df['close'].pct_change(-1))
            
        elif target_type == 'next_day_price':
            # Predict the actual price
            df['target'] = df['close'].shift(-1)
            
        elif target_type == 'multi_day_return':
            # Predict the return over multiple days
            df['target'] = df['close'].pct_change(-horizon)
            
        elif target_type == 'volatility':
            # Predict volatility
            df['target'] = df['returns_1d'].rolling(window=horizon).std() * np.sqrt(252)
            df['target'] = df['target'].shift(-horizon)  # Align with prediction time
            
        return df
    
    def _train_symbol_model(self, 
                           data: pd.DataFrame,
                           model_type: str,
                           test_size: float,
                           cv_folds: int) -> Tuple[Any, Dict[str, float]]:
        """
        Train a model for a specific symbol.
        
        Args:
            data: DataFrame with features and target
            model_type: Type of model to train
            test_size: Proportion of data to use for testing
            cv_folds: Number of folds for cross-validation
            
        Returns:
            Tuple of (trained model, performance metrics)
        """
        # Drop rows with NaN values
        data = data.dropna()
        
        # Check if we have enough data
        if len(data) < 100:
            logger.warning(f"Not enough data for reliable model training: {len(data)} rows")
            return None, {}
        
        # Define features and target
        feature_cols = [col for col in data.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
        X = data[feature_cols]
        y = data['target']
        
        # Create train/test split
        split_idx = int(len(data) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create model
        model_func = self.model_types.get(model_type, self._create_xgboost)
        model = model_func()
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train_scaled):
            # Split the data
            cv_X_train, cv_X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            cv_y_train, cv_y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Fit on training data
            model.fit(cv_X_train, cv_y_train)
            
            # Predict and evaluate
            cv_y_pred = model.predict(cv_X_val)
            cv_scores.append(mean_squared_error(cv_y_val, cv_y_pred))
        
        # Train final model on all training data
        model.fit(X_train_scaled, y_train)
        
        # Make predictions on test set
        y_pred = model.predict(X_test_scaled)
        
        # Calculate performance metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # For direction prediction, calculate accuracy
        if y.dtype in [bool, int, np.int64] and set(y.unique()).issubset({-1, 0, 1}):
            direction_accuracy = np.mean((np.sign(y_pred) == y_test).astype(int))
        else:
            direction_accuracy = np.mean((np.sign(y_pred) == np.sign(y_test)).astype(int))
        
        # Calculate profitability (simplified)
        # Assuming perfect execution at predicted values
        pred_returns = y_pred * np.sign(y_pred)  # Only take trades in predicted direction
        profit_factor = np.sum(pred_returns[pred_returns > 0]) / abs(np.sum(pred_returns[pred_returns < 0])) if np.sum(pred_returns[pred_returns < 0]) < 0 else np.inf
        
        # Store performance metrics
        performance = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'cv_mse_mean': np.mean(cv_scores),
            'cv_mse_std': np.std(cv_scores),
            'direction_accuracy': direction_accuracy,
            'profit_factor': profit_factor
        }
        
        # Feature importance
        try:
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            performance['feature_importance'] = importance_df.to_dict('records')
        except:
            logger.info("Feature importance not available for this model type")
        
        return model, performance
    
    def _plot_performance(self, 
                         symbol: str, 
                         data: pd.DataFrame, 
                         model: Any, 
                         performance: Dict[str, float]) -> None:
        """
        Generate and save performance visualizations.
        
        Args:
            symbol: Stock symbol
            data: DataFrame with features and target
            model: Trained model
            performance: Dictionary of performance metrics
        """
        try:
            # Create figure directory
            figures_dir = 'results/figures'
            os.makedirs(figures_dir, exist_ok=True)
            
            # Define features
            feature_cols = [col for col in data.columns if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
            
            # Define test set
            test_size = 0.2
            split_idx = int(len(data) * (1 - test_size))
            test_data = data.iloc[split_idx:]
            
            # Get model predictions for test data
            X_test = test_data[feature_cols].dropna()
            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(X_test)
            y_test = test_data.loc[X_test.index, 'target']
            y_pred = model.predict(X_test_scaled)
            
            # Create plot
            fig, axs = plt.subplots(3, 1, figsize=(12, 15))
            
            # Plot 1: Actual vs Predicted
            axs[0].plot(y_test.index, y_test.values, label='Actual')
            axs[0].plot(y_test.index, y_pred, label='Predicted')
            axs[0].set_title(f'{symbol} - Actual vs Predicted Values')
            axs[0].legend()
            axs[0].grid(True)
            
            # Plot 2: Cumulative Returns
            if 'close' in test_data.columns:
                test_data['strategy_returns'] = np.sign(y_pred) * test_data['close'].pct_change()
                test_data['buy_hold_returns'] = test_data['close'].pct_change()
                
                cum_strategy = (1 + test_data['strategy_returns'].fillna(0)).cumprod()
                cum_buyhold = (1 + test_data['buy_hold_returns'].fillna(0)).cumprod()
                
                axs[1].plot(cum_strategy.index, cum_strategy, label='Model Strategy')
                axs[1].plot(cum_buyhold.index, cum_buyhold, label='Buy & Hold')
                axs[1].set_title(f'{symbol} - Cumulative Returns')
                axs[1].legend()
                axs[1].grid(True)
            
            # Plot 3: Feature Importance
            if 'feature_importance' in performance:
                importance_df = pd.DataFrame(performance['feature_importance'])
                top_features = importance_df.head(15)
                axs[2].barh(top_features['feature'], top_features['importance'])
                axs[2].set_title(f'{symbol} - Top 15 Feature Importance')
                axs[2].grid(True)
            
            # Display performance metrics on the plot
            performance_text = f"MSE: {performance.get('mse', 0):.6f}\n"
            performance_text += f"Direction Accuracy: {performance.get('direction_accuracy', 0):.2%}\n"
            performance_text += f"Profit Factor: {performance.get('profit_factor', 0):.2f}\n"
            
            fig.text(0.1, 0.01, performance_text, fontsize=12)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f'{symbol}_model_performance.png'))
            plt.close()
            
            logger.info(f"Performance visualization saved for {symbol}")
        except Exception as e:
            logger.error(f"Error creating performance visualization: {e}")
    
    def _create_random_forest(self):
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    
    def _create_gradient_boosting(self):
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
    def _create_xgboost(self):
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=50,  # Reduced from 100
            learning_rate=0.1,
            max_depth=3,     # Reduced from 5
            random_state=42,
            n_jobs=2,        # Limit parallel jobs
            subsample=0.8,   # Use 80% of data for training
            colsample_bytree=0.8  # Use 80% of features
        )
    
    def _create_ridge(self):
        return Ridge(alpha=1.0)


def train_model(config=None):
    """
    Train a model using the ModelTrainer class.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Trained model
    """
    trainer = ModelTrainer()
    return trainer.train_model(config)


if __name__ == "__main__":
    # Example configuration
    example_config = {
        'model_type': 'xgboost',
        'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
        'prediction_target': 'next_day_return',
        'features': ['rsi', 'macd', 'bollinger', 'volume', 'momentum', 'trend'],
        'run_backtest': True
    }
    
    # Train model with example configuration
    trained_model = train_model(example_config)
    
    # Print performance summary
    for symbol, perf in trained_model['performance'].items():
        print(f"\n{symbol} Performance:")
        print(f"  Direction Accuracy: {perf.get('direction_accuracy', 0):.2%}")
        print(f"  Profit Factor: {perf.get('profit_factor', 0):.2f}")
