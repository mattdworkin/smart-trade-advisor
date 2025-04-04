import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Detects market regimes based on various market indicators.
    Regimes: bullish, bearish, volatile, neutral
    """

    def __init__(self):
        self.current_regime = "neutral"

    def detect_regime(self, market_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Detect the current market regime based on indicators

        Args:
            market_data: Dictionary with market data (if None, uses default values)

        Returns:
            Regime classification: "bullish", "bearish", "volatile", or "neutral" 
        """
        try:
            # If no data provided, attempt to detect based on general market conditions
            # This is a placeholder implementation
            # In a real system, you would analyze VIX, market breadth, moving averages, etc.

            if market_data is None:
                # Return a random regime for demo purposes
                # In production, you would connect to real market data
                import random
                regimes = ["bullish", "bearish", "volatile", "neutral"]
                weights = [0.3, 0.2, 0.2, 0.3]  # Weights for randomization
                regime = random.choices(regimes, weights=weights, k=1)[0]

                # Log the regime change if it's different
                if regime != self.current_regime:
                    logger.info(f"Market regime changed from {self.current_regime} to {regime}")
                    self.current_regime = regime

                return regime

            # With actual market data, implement a more sophisticated approach
            # Example: volatility-based regime detection
            if "vix" in market_data:
                vix = market_data["vix"]
                if vix > 30:
                    return "volatile"
                elif vix < 15:
                    return "bullish"

            # Example: trend-based regime detection  
            if "sp500_ma_crossover" in market_data:
                crossover = market_data["sp500_ma_crossover"]
                if crossover > 0:
                    return "bullish"
                elif crossover < 0:
                    return "bearish"

            # Default to neutral if no conditions are met
            return "neutral"

        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return "neutral"  # Default to neutral on error
