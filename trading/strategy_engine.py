import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class StrategyEngine:
    """
    Implements various trading strategies and generates trade suggestions
    based on market data.
    """
    
    def __init__(self, confidence_threshold: float = 0.65):
        """
        Initialize the strategy engine.
        
        Args:
            confidence_threshold: Minimum confidence level for trade suggestions
        """
        self.confidence_threshold = confidence_threshold
        self.strategies = {
            "moving_average_crossover": self._moving_average_crossover,
            "rsi_divergence": self._rsi_strategy,
            "volume_breakout": self._volume_breakout,
            "trend_following": self._trend_following,
            "mean_reversion": self._mean_reversion
        }
        
    def generate_suggestions(self, 
                            historical_data: Dict[str, pd.DataFrame], 
                            portfolio: Dict[str, Any],
                            market_regime: str = "neutral") -> List[Dict[str, Any]]:
        """
        Generate trade suggestions based on available strategies and market data.
        
        Args:
            historical_data: Dictionary of symbol -> DataFrame with historical price data
            portfolio: Current portfolio holdings
            market_regime: Current market regime (bullish, bearish, neutral, volatile)
            
        Returns:
            List of trade suggestions
        """
        all_suggestions = []
        
        # Run each strategy
        for strategy_name, strategy_func in self.strategies.items():
            try:
                # Skip strategies that don't align with the current market regime
                if not self._is_strategy_suitable(strategy_name, market_regime):
                    continue
                
                strategy_suggestions = strategy_func(historical_data, portfolio)
                
                # Add strategy name to each suggestion
                for suggestion in strategy_suggestions:
                    suggestion["strategy"] = strategy_name
                
                all_suggestions.extend(strategy_suggestions)
            except Exception as e:
                logger.error(f"Error running strategy {strategy_name}: {e}")
        
        # Filter suggestions by confidence threshold
        filtered_suggestions = [
            s for s in all_suggestions 
            if s.get("confidence", 0) >= self.confidence_threshold
        ]
        
        # Sort by confidence (highest first)
        filtered_suggestions.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Limit to reasonable number of suggestions
        return filtered_suggestions[:5]
    
    def _is_strategy_suitable(self, strategy_name: str, market_regime: str) -> bool:
        """
        Determine if a strategy is suitable for the current market regime.
        
        Args:
            strategy_name: Name of the strategy
            market_regime: Current market regime
            
        Returns:
            True if the strategy is suitable for the current market regime
        """
        # Strategy suitability matrix
        suitability = {
            "moving_average_crossover": ["bullish", "bearish", "neutral"],
            "rsi_divergence": ["volatile", "neutral"],
            "volume_breakout": ["volatile", "bullish"],
            "trend_following": ["bullish", "bearish"],
            "mean_reversion": ["volatile", "neutral"]
        }
        
        return market_regime in suitability.get(strategy_name, [])
        
    def _moving_average_crossover(self, 
                                historical_data: Dict[str, pd.DataFrame],
                                portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate suggestions based on moving average crossover strategy.
        Buy when short MA crosses above long MA, sell when it crosses below.
        
        Args:
            historical_data: Dictionary of symbol -> DataFrame with historical price data
            portfolio: Current portfolio holdings
            
        Returns:
            List of trade suggestions
        """
        suggestions = []
        portfolio_positions = portfolio.get("positions", {})
        
        for symbol, data in historical_data.items():
            if len(data) < 50:  # Need enough data for moving averages
                continue
                
            # Calculate short and long moving averages
            data['MA20'] = data['close'].rolling(window=20).mean()
            data['MA50'] = data['close'].rolling(window=50).mean()
            
            # Get the last few days of data
            last_row = data.iloc[-1]
            prev_row = data.iloc[-2]
            
            # Check for crossover events
            if prev_row['MA20'] <= prev_row['MA50'] and last_row['MA20'] > last_row['MA50']:
                # Bullish crossover (buy signal)
                
                # Calculate confidence based on the strength of the crossover
                crossover_strength = (last_row['MA20'] - last_row['MA50']) / last_row['MA50']
                confidence = min(0.9, 0.65 + crossover_strength * 100)
                
                # Calculate position size based on confidence
                position_value = portfolio.get("cash", 0) * 0.1 * confidence
                current_price = last_row['close']
                quantity = max(1, int(position_value / current_price))
                
                suggestions.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": quantity,
                    "confidence": confidence,
                    "reason": f"Bullish moving average crossover: 20-day MA crossed above 50-day MA"
                })
                
            elif prev_row['MA20'] >= prev_row['MA50'] and last_row['MA20'] < last_row['MA50']:
                # Bearish crossover (sell signal)
                
                # Check if we hold this stock
                position = portfolio_positions.get(symbol)
                if position and position.get("shares", 0) > 0:
                    # Calculate confidence based on the strength of the crossover
                    crossover_strength = (last_row['MA50'] - last_row['MA20']) / last_row['MA50']
                    confidence = min(0.9, 0.65 + crossover_strength * 100)
                    
                    # Suggest selling a portion or all based on confidence
                    sell_portion = 0.5 + (confidence - 0.65) * 2  # 50% to 100%
                    quantity = max(1, int(position.get("shares", 0) * sell_portion))
                    
                    suggestions.append({
                        "symbol": symbol,
                        "action": "SELL",
                        "quantity": quantity,
                        "confidence": confidence,
                        "reason": f"Bearish moving average crossover: 20-day MA crossed below 50-day MA"
                    })
        
        return suggestions
    
    def _rsi_strategy(self, 
                    historical_data: Dict[str, pd.DataFrame],
                    portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate suggestions based on RSI (Relative Strength Index) strategy.
        Buy when oversold, sell when overbought.
        
        Args:
            historical_data: Dictionary of symbol -> DataFrame with historical price data
            portfolio: Current portfolio holdings
            
        Returns:
            List of trade suggestions
        """
        suggestions = []
        portfolio_positions = portfolio.get("positions", {})
        
        for symbol, data in historical_data.items():
            if len(data) < 15:  # Need enough data for RSI
                continue
                
            # Calculate RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            
            # Calculate RS and RSI
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            last_rsi = data['RSI'].iloc[-1]
            current_price = data['close'].iloc[-1]
            
            # Oversold condition (RSI < 30) - Buy signal
            if last_rsi < 30:
                # Calculate confidence based on how oversold
                confidence = 0.65 + min(0.25, (30 - last_rsi) / 50)
                
                # Calculate position size based on confidence
                position_value = portfolio.get("cash", 0) * 0.1 * confidence
                quantity = max(1, int(position_value / current_price))
                
                suggestions.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": quantity,
                    "confidence": confidence,
                    "reason": f"Oversold condition: RSI at {last_rsi:.1f} (below 30)"
                })
                
            # Overbought condition (RSI > 70) - Sell signal
            elif last_rsi > 70:
                # Check if we hold this stock
                position = portfolio_positions.get(symbol)
                if position and position.get("shares", 0) > 0:
                    # Calculate confidence based on how overbought
                    confidence = 0.65 + min(0.25, (last_rsi - 70) / 30)
                    
                    # Suggest selling a portion based on confidence
                    sell_portion = 0.5 + (confidence - 0.65) * 2  # 50% to 100%
                    quantity = max(1, int(position.get("shares", 0) * sell_portion))
                    
                    suggestions.append({
                        "symbol": symbol,
                        "action": "SELL",
                        "quantity": quantity,
                        "confidence": confidence,
                        "reason": f"Overbought condition: RSI at {last_rsi:.1f} (above 70)"
                    })
        
        return suggestions
    
    def _volume_breakout(self, 
                       historical_data: Dict[str, pd.DataFrame],
                       portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate suggestions based on volume breakout strategy.
        
        Args:
            historical_data: Dictionary of symbol -> DataFrame with historical price data
            portfolio: Current portfolio holdings
            
        Returns:
            List of trade suggestions
        """
        suggestions = []
        
        for symbol, data in historical_data.items():
            if len(data) < 30:  # Need enough data for volume analysis
                continue
                
            # Calculate volume moving average
            data['volume_ma20'] = data['volume'].rolling(window=20).mean()
            
            # Get the last few days of data
            last_row = data.iloc[-1]
            
            # Volume breakout condition - Buy signal
            if (last_row['volume'] > 2 * last_row['volume_ma20'] and
                last_row['close'] > last_row['close'] * 1.02):  # 2% price increase
                
                # Calculate confidence based on volume increase
                volume_ratio = last_row['volume'] / last_row['volume_ma20']
                confidence = min(0.9, 0.65 + (volume_ratio - 2) * 0.1)
                
                # Calculate position size based on confidence
                position_value = portfolio.get("cash", 0) * 0.1 * confidence
                current_price = last_row['close']
                quantity = max(1, int(position_value / current_price))
                
                suggestions.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": quantity,
                    "confidence": confidence,
                    "reason": f"Volume breakout: {volume_ratio:.1f}x average volume with price increase"
                })
        
        return suggestions
    
    def _trend_following(self, 
                       historical_data: Dict[str, pd.DataFrame],
                       portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate suggestions based on trend following strategy.
        
        Args:
            historical_data: Dictionary of symbol -> DataFrame with historical price data
            portfolio: Current portfolio holdings
            
        Returns:
            List of trade suggestions
        """
        suggestions = []
        portfolio_positions = portfolio.get("positions", {})
        
        for symbol, data in historical_data.items():
            if len(data) < 60:  # Need enough data
                continue
                
            # Calculate indicators
            data['MA50'] = data['close'].rolling(window=50).mean()
            data['MA20'] = data['close'].rolling(window=20).mean()
            
            # Get the last data point
            last_row = data.iloc[-1]
            last_close = last_row['close']
            last_ma20 = last_row['MA20']
            last_ma50 = last_row['MA50']
            
            # Calculate the slope of MA20 (simple calculation)
            ma20_slope = (last_ma20 - data['MA20'].iloc[-10]) / 10
            
            # Strong uptrend conditions - Buy signal
            if (last_close > last_ma20 > last_ma50 and 
                ma20_slope > 0):
                
                # Calculate confidence based on slope steepness
                confidence = min(0.9, 0.65 + ma20_slope * 200)  # Scale slope to confidence
                
                # Calculate position size based on confidence
                position_value = portfolio.get("cash", 0) * 0.1 * confidence
                quantity = max(1, int(position_value / last_close))
                
                suggestions.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": quantity,
                    "confidence": confidence,
                    "reason": f"Strong uptrend: Price above both moving averages with positive slope"
                })
                
            # Strong downtrend conditions - Sell signal
            elif (last_close < last_ma20 < last_ma50 and 
                  ma20_slope < 0):
                
                # Check if we hold this stock
                position = portfolio_positions.get(symbol)
                if position and position.get("shares", 0) > 0:
                    # Calculate confidence based on slope steepness
                    confidence = min(0.9, 0.65 + abs(ma20_slope) * 200)
                    
                    # Suggest selling portion based on confidence
                    sell_portion = 0.5 + (confidence - 0.65) * 2  # 50% to 100%
                    quantity = max(1, int(position.get("shares", 0) * sell_portion))
                    
                    suggestions.append({
                        "symbol": symbol,
                        "action": "SELL",
                        "quantity": quantity,
                        "confidence": confidence,
                        "reason": f"Strong downtrend: Price below both moving averages with negative slope"
                    })
        
        return suggestions
    
    def _mean_reversion(self, 
                      historical_data: Dict[str, pd.DataFrame],
                      portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate suggestions based on mean reversion strategy.
        
        Args:
            historical_data: Dictionary of symbol -> DataFrame with historical price data
            portfolio: Current portfolio holdings
            
        Returns:
            List of trade suggestions
        """
        suggestions = []
        portfolio_positions = portfolio.get("positions", {})
        
        for symbol, data in historical_data.items():
            if len(data) < 60:  # Need enough data
                continue
                
            # Calculate moving average and bollinger bands
            data['MA20'] = data['close'].rolling(window=20).mean()
            data['std20'] = data['close'].rolling(window=20).std()
            data['upper_band'] = data['MA20'] + 2 * data['std20']
            data['lower_band'] = data['MA20'] - 2 * data['std20']
            
            # Get the last data point
            last_row = data.iloc[-1]
            last_close = last_row['close']
            
            # Price below lower band - Buy signal (expect reversion to mean)
            if last_close < last_row['lower_band']:
                # Calculate how far below the band (for confidence calculation)
                deviation = (last_row['lower_band'] - last_close) / last_row['std20']
                confidence = min(0.9, 0.65 + deviation * 0.1)
                
                # Calculate position size based on confidence
                position_value = portfolio.get("cash", 0) * 0.1 * confidence
                quantity = max(1, int(position_value / last_close))
                
                suggestions.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": quantity,
                    "confidence": confidence,
                    "reason": f"Mean reversion: Price below lower Bollinger Band by {deviation:.2f} standard deviations"
                })
                
            # Price above upper band - Sell signal (expect reversion to mean)
            elif last_close > last_row['upper_band']:
                # Check if we hold this stock
                position = portfolio_positions.get(symbol)
                if position and position.get("shares", 0) > 0:
                    # Calculate how far above the band (for confidence calculation)
                    deviation = (last_close - last_row['upper_band']) / last_row['std20']
                    confidence = min(0.9, 0.65 + deviation * 0.1)
                    
                    # Suggest selling portion based on confidence
                    sell_portion = 0.5 + (confidence - 0.65) * 2  # 50% to 100%
                    quantity = max(1, int(position.get("shares", 0) * sell_portion))
                    
                    suggestions.append({
                        "symbol": symbol,
                        "action": "SELL",
                        "quantity": quantity,
                        "confidence": confidence,
                        "reason": f"Mean reversion: Price above upper Bollinger Band by {deviation:.2f} standard deviations"
                    })
        
        return suggestions 