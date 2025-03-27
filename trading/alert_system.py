import pandas as pd
import numpy as np
import logging
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading
import queue

logger = logging.getLogger(__name__)

class AlertSystem:
    """
    Real-time alert system for intraday pattern detection
    """
    
    def __init__(self, data_fetcher=None, realtime_stream=None):
        """
        Initialize the alert system
        
        Args:
            data_fetcher: Historical data fetcher instance
            realtime_stream: Realtime data stream instance
        """
        self.data_fetcher = data_fetcher
        self.realtime_stream = realtime_stream
        self.alerts_dir = 'results/alerts'
        os.makedirs(self.alerts_dir, exist_ok=True)
        
        # Alert settings
        self.alert_config = {
            'volume_surge': {
                'enabled': True,
                'threshold': 2.0,  # Volume vs average
                'lookback': 10,    # Periods to compare against
                'cooldown': 30     # Minutes between alerts for same symbol
            },
            'price_breakout': {
                'enabled': True,
                'threshold': 1.5,  # Percent move in short time
                'lookback': 20,    # Candles to establish range
                'cooldown': 30
            },
            'rsi_extreme': {
                'enabled': True,
                'overbought': 70,
                'oversold': 30,
                'cooldown': 30
            },
            'moving_average_cross': {
                'enabled': True,
                'fast_period': 5,
                'slow_period': 20,
                'cooldown': 30
            }
        }
        
        # Track alerts to avoid duplicates
        self.alert_history = {}
        
        # Thread management
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.alert_queue = queue.Queue()
        
        # Callbacks for alert handling
        self.alert_callbacks = []
        
        # Symbols being monitored
        self.monitored_symbols = []
    
    def start_monitoring(self, symbols: List[str], interval: str = '5m') -> None:
        """
        Start monitoring symbols for alerts
        
        Args:
            symbols: List of symbols to monitor
            interval: Data interval for monitoring
        """
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already active, stop first before restarting")
            return
        
        logger.info(f"Starting alert monitoring for {len(symbols)} symbols with {interval} interval")
        
        # Reset stop flag
        self.stop_monitoring.clear()
        
        # Store symbols
        self.monitored_symbols = symbols
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(symbols, interval),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Alert monitoring started")
    
    def stop_monitoring_alerts(self) -> None:
        """Stop the alert monitoring thread"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.info("Stopping alert monitoring...")
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5)
            logger.info("Alert monitoring stopped")
    
    def add_alert_callback(self, callback_function) -> None:
        """
        Add a callback function to handle alerts
        
        Args:
            callback_function: Function that takes an alert dict as parameter
        """
        self.alert_callbacks.append(callback_function)
        logger.info(f"Added alert callback, total callbacks: {len(self.alert_callbacks)}")
    
    def get_pending_alerts(self) -> List[Dict[str, Any]]:
        """
        Get all pending alerts from the queue
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        while not self.alert_queue.empty():
            try:
                alerts.append(self.alert_queue.get_nowait())
                self.alert_queue.task_done()
            except queue.Empty:
                break
        
        return alerts
    
    def check_for_patterns(self, symbol: str, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Check a symbol's data for alert patterns
        
        Args:
            symbol: Symbol to check
            data: DataFrame with OHLCV data
            
        Returns:
            List of detected alert patterns
        """
        if data is None or data.empty:
            return []
        
        alerts = []
        
        # Make sure data has the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            logger.warning(f"Data for {symbol} missing required columns")
            return []
        
        # Check each enabled alert type
        current_time = datetime.now()
        
        # 1. Volume Surge alert
        if self.alert_config['volume_surge']['enabled']:
            # Check if cooldown has passed
            cooldown_key = f"{symbol}_volume_surge"
            cooldown_passed = self._check_cooldown(cooldown_key, self.alert_config['volume_surge']['cooldown'])
            
            if cooldown_passed:
                # Calculate average volume
                lookback = self.alert_config['volume_surge']['lookback']
                if len(data) >= lookback + 1:
                    recent_volume = data['volume'].iloc[-1]
                    avg_volume = data['volume'].iloc[-(lookback+1):-1].mean()
                    
                    # Check if current volume exceeds threshold
                    if avg_volume > 0 and recent_volume / avg_volume >= self.alert_config['volume_surge']['threshold']:
                        alerts.append({
                            'symbol': symbol,
                            'alert_type': 'volume_surge',
                            'time': current_time,
                            'data': {
                                'recent_volume': int(recent_volume),
                                'avg_volume': int(avg_volume),
                                'volume_ratio': recent_volume / avg_volume,
                                'price': data['close'].iloc[-1]
                            }
                        })
                        
                        # Update cooldown
                        self.alert_history[cooldown_key] = current_time
        
        # 2. Price Breakout alert
        if self.alert_config['price_breakout']['enabled']:
            cooldown_key = f"{symbol}_price_breakout"
            cooldown_passed = self._check_cooldown(cooldown_key, self.alert_config['price_breakout']['cooldown'])
            
            if cooldown_passed:
                lookback = self.alert_config['price_breakout']['lookback']
                if len(data) >= lookback + 1:
                    # Calculate recent high/low range
                    recent_high = data['high'].iloc[-(lookback+1):-1].max()
                    recent_low = data['low'].iloc[-(lookback+1):-1].min()
                    range_size = recent_high - recent_low
                    
                    current_price = data['close'].iloc[-1]
                    prev_price = data['close'].iloc[-2]
                    
                    # Breakout above recent high
                    if (current_price > recent_high and 
                            (current_price - prev_price) / prev_price * 100 >= self.alert_config['price_breakout']['threshold']):
                        alerts.append({
                            'symbol': symbol,
                            'alert_type': 'price_breakout_up',
                            'time': current_time,
                            'data': {
                                'price': current_price,
                                'breakout_level': recent_high,
                                'percent_change': (current_price - prev_price) / prev_price * 100,
                                'volume': data['volume'].iloc[-1]
                            }
                        })
                        
                        # Update cooldown
                        self.alert_history[cooldown_key] = current_time
                    
                    # Breakdown below recent low
                    elif (current_price < recent_low and 
                            (prev_price - current_price) / prev_price * 100 >= self.alert_config['price_breakout']['threshold']):
                        alerts.append({
                            'symbol': symbol,
                            'alert_type': 'price_breakout_down',
                            'time': current_time,
                            'data': {
                                'price': current_price,
                                'breakdown_level': recent_low,
                                'percent_change': (current_price - prev_price) / prev_price * 100,
                                'volume': data['volume'].iloc[-1]
                            }
                        })
                        
                        # Update cooldown
                        self.alert_history[cooldown_key] = current_time
        
        # 3. RSI Extreme alert
        if self.alert_config['rsi_extreme']['enabled']:
            cooldown_key = f"{symbol}_rsi_extreme"
            cooldown_passed = self._check_cooldown(cooldown_key, self.alert_config['rsi_extreme']['cooldown'])
            
            if cooldown_passed:
                # Calculate RSI
                try:
                    from models.model_trainer import ModelTrainer
                    model_trainer = ModelTrainer()
                    
                    # Use the engineer_features method to calculate RSI
                    feature_data = model_trainer._engineer_features(
                        data,
                        ['rsi'],
                        14  # Standard RSI period
                    )
                    
                    if 'rsi_14' in feature_data.columns:
                        current_rsi = feature_data['rsi_14'].iloc[-1]
                        
                        # Check for overbought condition
                        if current_rsi > self.alert_config['rsi_extreme']['overbought']:
                            alerts.append({
                                'symbol': symbol,
                                'alert_type': 'rsi_overbought',
                                'time': current_time,
                                'data': {
                                    'rsi': current_rsi,
                                    'price': data['close'].iloc[-1],
                                    'overbought_level': self.alert_config['rsi_extreme']['overbought']
                                }
                            })
                            
                            # Update cooldown
                            self.alert_history[cooldown_key] = current_time
                        
                        # Check for oversold condition
                        elif current_rsi < self.alert_config['rsi_extreme']['oversold']:
                            alerts.append({
                                'symbol': symbol,
                                'alert_type': 'rsi_oversold',
                                'time': current_time,
                                'data': {
                                    'rsi': current_rsi,
                                    'price': data['close'].iloc[-1],
                                    'oversold_level': self.alert_config['rsi_extreme']['oversold']
                                }
                            })
                            
                            # Update cooldown
                            self.alert_history[cooldown_key] = current_time
                except Exception as e:
                    logger.warning(f"Error calculating RSI for {symbol}: {e}")
        
        # 4. Moving Average Cross alert
        if self.alert_config['moving_average_cross']['enabled']:
            cooldown_key = f"{symbol}_ma_cross"
            cooldown_passed = self._check_cooldown(cooldown_key, self.alert_config['moving_average_cross']['cooldown'])
            
            if cooldown_passed:
                fast_period = self.alert_config['moving_average_cross']['fast_period']
                slow_period = self.alert_config['moving_average_cross']['slow_period']
                
                if len(data) >= slow_period + 1:
                    # Calculate moving averages
                    fast_ma = data['close'].rolling(window=fast_period).mean()
                    slow_ma = data['close'].rolling(window=slow_period).mean()
                    
                    # Check for crossover (fast MA crosses above slow MA)
                    if fast_ma.iloc[-2] <= slow_ma.iloc[-2] and fast_ma.iloc[-1] > slow_ma.iloc[-1]:
                        alerts.append({
                            'symbol': symbol,
                            'alert_type': 'ma_cross_bullish',
                            'time': current_time,
                            'data': {
                                'fast_ma': fast_ma.iloc[-1],
                                'slow_ma': slow_ma.iloc[-1],
                                'price': data['close'].iloc[-1],
                                'fast_period': fast_period,
                                'slow_period': slow_period
                            }
                        })
                        
                        # Update cooldown
                        self.alert_history[cooldown_key] = current_time
                    
                    # Check for crossunder (fast MA crosses below slow MA)
                    elif fast_ma.iloc[-2] >= slow_ma.iloc[-2] and fast_ma.iloc[-1] < slow_ma.iloc[-1]:
                        alerts.append({
                            'symbol': symbol,
                            'alert_type': 'ma_cross_bearish',
                            'time': current_time,
                            'data': {
                                'fast_ma': fast_ma.iloc[-1],
                                'slow_ma': slow_ma.iloc[-1],
                                'price': data['close'].iloc[-1],
                                'fast_period': fast_period,
                                'slow_period': slow_period
                            }
                        })
                        
                        # Update cooldown
                        self.alert_history[cooldown_key] = current_time
        
        # Log alerts
        if alerts:
            for alert in alerts:
                logger.info(f"Alert triggered: {alert['alert_type']} for {symbol}")
                self._log_alert(alert)
                
                # Add to alert queue for notification
                self.alert_queue.put(alert)
                
                # Trigger callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")
        
        return alerts
    
    def check_all_symbols(self, interval: str = '5m') -> List[Dict[str, Any]]:
        """
        Check all monitored symbols for alerts
        
        Args:
            interval: Data interval to check
            
        Returns:
            List of all detected alerts
        """
        if not self.monitored_symbols:
            logger.warning("No symbols being monitored for alerts")
            return []
        
        all_alerts = []
        
        for symbol in self.monitored_symbols:
            try:
                # Get data for the symbol
                if self.data_fetcher:
                    data = None
                    if interval.endswith('m'):
                        # Intraday data
                        data = self.data_fetcher.fetch_intraday_data(symbol, interval=interval)
                    else:
                        # Daily data
                        data = self.data_fetcher.fetch_daily_data(symbol, days=50)
                    
                    if data is not None and not data.empty:
                        # Check for patterns
                        alerts = self.check_for_patterns(symbol, data)
                        all_alerts.extend(alerts)
                    else:
                        logger.warning(f"No data available for {symbol}")
                else:
                    logger.warning("No data fetcher available for alerts")
            except Exception as e:
                logger.error(f"Error checking alerts for {symbol}: {e}")
        
        return all_alerts
    
    def _monitoring_loop(self, symbols: List[str], interval: str) -> None:
        """
        Main monitoring loop that runs in a separate thread
        
        Args:
            symbols: List of symbols to monitor
            interval: Data interval for monitoring
        """
        logger.info(f"Alert monitoring loop started for {len(symbols)} symbols")
        
        try:
            # Calculate sleep time based on interval
            sleep_minutes = 5  # Default to 5 minutes
            
            if interval.endswith('m'):
                # Extract minutes from interval string (e.g., '5m' -> 5)
                try:
                    interval_minutes = int(interval[:-1])
                    sleep_minutes = max(1, interval_minutes // 2)  # Half the interval time, minimum 1 minute
                except ValueError:
                    pass
            elif interval.endswith('h'):
                # Extract hours from interval string
                try:
                    interval_hours = int(interval[:-1])
                    sleep_minutes = interval_hours * 30  # 30 minutes for hourly data
                except ValueError:
                    pass
            
            sleep_seconds = sleep_minutes * 60
            
            # Initial check
            self.check_all_symbols(interval)
            
            # Monitoring loop
            while not self.stop_monitoring.is_set():
                # Sleep for the specified time
                for _ in range(0, sleep_seconds, 5):
                    if self.stop_monitoring.is_set():
                        break
                    time.sleep(5)
                
                # Check all symbols
                if not self.stop_monitoring.is_set():
                    self.check_all_symbols(interval)
        
        except Exception as e:
            logger.error(f"Error in alert monitoring loop: {e}")
        
        logger.info("Alert monitoring loop ended")
    
    def _check_cooldown(self, key: str, cooldown_minutes: int) -> bool:
        """
        Check if cooldown period has passed for a given alert key
        
        Args:
            key: Alert history key
            cooldown_minutes: Cooldown period in minutes
            
        Returns:
            True if cooldown has passed, False otherwise
        """
        if key not in self.alert_history:
            return True
        
        last_alert_time = self.alert_history[key]
        cooldown_delta = timedelta(minutes=cooldown_minutes)
        
        return datetime.now() - last_alert_time > cooldown_delta
    
    def _log_alert(self, alert: Dict[str, Any]) -> None:
        """
        Log an alert to a file for historical reference
        
        Args:
            alert: Alert dictionary
        """
        try:
            # Create log file name based on date
            log_file = os.path.join(self.alerts_dir, f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl")
            
            # Format the alert time as a string
            if 'time' in alert and isinstance(alert['time'], datetime):
                alert = alert.copy()  # Make a copy to avoid modifying the original
                alert['time'] = alert['time'].isoformat()
            
            # Append to log file
            with open(log_file, 'a') as f:
                f.write(json.dumps(alert) + '\n')
        except Exception as e:
            logger.error(f"Error logging alert: {e}")
    
    def get_alert_history(self, symbol: Optional[str] = None, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get historical alerts
        
        Args:
            symbol: Optional symbol to filter alerts
            days: Number of days of history to retrieve
            
        Returns:
            List of historical alerts
        """
        history = []
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get list of alert log files in date range
            for day in range(days):
                date = end_date - timedelta(days=day)
                log_file = os.path.join(self.alerts_dir, f"alerts_{date.strftime('%Y%m%d')}.jsonl")
                
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        for line in f:
                            try:
                                alert = json.loads(line.strip())
                                
                                # Parse alert time
                                if 'time' in alert and isinstance(alert['time'], str):
                                    alert['time'] = datetime.fromisoformat(alert['time'])
                                
                                # Filter by symbol if specified
                                if symbol is None or alert.get('symbol') == symbol:
                                    history.append(alert)
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.error(f"Error retrieving alert history: {e}")
        
        # Sort by time (newest first)
        history.sort(key=lambda x: x.get('time', datetime.min), reverse=True)
        
        return history
    
    def find_potential_trades(self) -> List[Dict[str, Any]]:
        """
        Find potential trades based on recent alerts
        
        Returns:
            List of potential trade setups
        """
        # Get alerts from the last 24 hours
        recent_alerts = self.get_alert_history(days=1)
        
        potential_trades = []
        processed_symbols = set()
        
        # Process alerts in order of recency
        for alert in recent_alerts:
            symbol = alert.get('symbol')
            
            # Skip if we've already identified a trade for this symbol
            if symbol in processed_symbols:
                continue
            
            alert_type = alert.get('alert_type', '')
            
            # Different trade setups based on alert type
            if 'price_breakout_up' in alert_type:
                potential_trades.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'setup': 'Breakout',
                    'alert_time': alert.get('time'),
                    'price': alert.get('data', {}).get('price', 0),
                    'stop_loss': alert.get('data', {}).get('breakout_level', 0) * 0.97,  # 3% below breakout level
                    'confidence': 0.7
                })
                processed_symbols.add(symbol)
                
            elif 'rsi_oversold' in alert_type:
                potential_trades.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'setup': 'Oversold Bounce',
                    'alert_time': alert.get('time'),
                    'price': alert.get('data', {}).get('price', 0),
                    'stop_loss': alert.get('data', {}).get('price', 0) * 0.95,  # 5% below current price
                    'confidence': 0.6
                })
                processed_symbols.add(symbol)
                
            elif 'ma_cross_bullish' in alert_type:
                potential_trades.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'setup': 'Bullish MA Cross',
                    'alert_time': alert.get('time'),
                    'price': alert.get('data', {}).get('price', 0),
                    'stop_loss': alert.get('data', {}).get('slow_ma', 0) * 0.97,  # 3% below slow MA
                    'confidence': 0.65
                })
                processed_symbols.add(symbol)
                
            elif 'rsi_overbought' in alert_type:
                potential_trades.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'setup': 'Overbought',
                    'alert_time': alert.get('time'),
                    'price': alert.get('data', {}).get('price', 0),
                    'confidence': 0.6
                })
                processed_symbols.add(symbol)
                
            elif 'ma_cross_bearish' in alert_type:
                potential_trades.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'setup': 'Bearish MA Cross',
                    'alert_time': alert.get('time'),
                    'price': alert.get('data', {}).get('price', 0),
                    'confidence': 0.65
                })
                processed_symbols.add(symbol)
        
        # Sort by confidence (highest first)
        potential_trades.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return potential_trades
    
    def update_alert_settings(self, new_settings: Dict[str, Any]) -> None:
        """
        Update alert settings
        
        Args:
            new_settings: Dictionary of new settings
        """
        # Update settings
        for alert_type, settings in new_settings.items():
            if alert_type in self.alert_config:
                for key, value in settings.items():
                    if key in self.alert_config[alert_type]:
                        self.alert_config[alert_type][key] = value
        
        logger.info("Alert settings updated") 