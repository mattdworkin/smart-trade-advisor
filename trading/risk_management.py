import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RiskMetrics:
    """Holds risk metrics for a portfolio or trade"""
    value_at_risk_95: float  # 95% VaR
    value_at_risk_99: float  # 99% VaR
    expected_shortfall: float  # Expected shortfall (CVaR)
    max_drawdown: float  # Maximum historical drawdown
    volatility: float  # Historical volatility
    sharpe_ratio: Optional[float] = None  # Sharpe ratio if return data available
    beta: Optional[float] = None  # Beta to benchmark if benchmark data available

class PositionSizing:
    """
    Determines appropriate position sizes based on risk parameters
    """
    
    @staticmethod
    def kelly_criterion(win_rate: float, win_loss_ratio: float, 
                        kelly_fraction: float = 0.5) -> float:
        """
        Calculate position size using the Kelly Criterion
        
        Args:
            win_rate: Probability of winning (0-1)
            win_loss_ratio: Ratio of average win to average loss
            kelly_fraction: Conservative adjustment (typically 0.5 or lower)
            
        Returns:
            Recommended position size as a fraction of capital
        """
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        # Apply the fraction to be more conservative
        conservative_kelly = kelly * kelly_fraction
        
        # Ensure we don't return negative values
        return max(0, conservative_kelly)
    
    @staticmethod
    def percent_risk_sizing(
        capital: float, 
        max_risk_percent: float,
        entry_price: float,
        stop_price: float
    ) -> float:
        """
        Size position based on percentage of capital at risk
        
        Args:
            capital: Total capital available
            max_risk_percent: Maximum percentage willing to risk (e.g., 0.01 for 1%)
            entry_price: Planned entry price
            stop_price: Planned stop loss price
            
        Returns:
            Number of shares/contracts to trade
        """
        if entry_price <= 0 or stop_price <= 0 or entry_price == stop_price:
            return 0
            
        risk_per_share = abs(entry_price - stop_price)
        max_risk_amount = capital * max_risk_percent
        position_size = max_risk_amount / risk_per_share
        
        return position_size

class RiskManager:
    """
    Main risk management system that enforces trading rules
    and monitors overall portfolio risk
    """
    
    def __init__(self, 
                 max_portfolio_risk: float = 0.02,  # 2% daily VaR
                 max_position_size: float = 0.15,   # 15% of portfolio
                 max_sector_exposure: float = 0.25, # 25% in any sector
                 max_correlation: float = 0.7):     # Max correlation between positions
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_correlation = max_correlation
        
    def calculate_var(self, 
                       returns: np.ndarray, 
                       confidence_level: float = 0.95, 
                       time_horizon: int = 1) -> float:
        """
        Calculate Value at Risk
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (typically 0.95 or 0.99)
            time_horizon: Time horizon in days
            
        Returns:
            Value at Risk as a positive number (representing loss)
        """
        if len(returns) < 100:
            # Not enough data for reliable VaR calculation
            return 0.0
            
        # Historical simulation method
        var = np.percentile(returns, 100 * (1 - confidence_level))
        
        # Scale by time horizon
        var = abs(var) * np.sqrt(time_horizon)
        
        return var
        
    def calculate_expected_shortfall(self, 
                                     returns: np.ndarray, 
                                     confidence_level: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level
            
        Returns:
            Expected shortfall as a positive number
        """
        var_cutoff = np.percentile(returns, 100 * (1 - confidence_level))
        beyond_var = returns[returns <= var_cutoff]
        
        if len(beyond_var) == 0:
            return abs(var_cutoff)
            
        return abs(beyond_var.mean())
    
    def analyze_portfolio_risk(self, 
                              positions: Dict[str, Dict],
                              historical_returns: Dict[str, np.ndarray],
                              correlation_matrix: Optional[pd.DataFrame] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for the entire portfolio
        
        Args:
            positions: Dictionary of current positions {symbol: {weight, sector, etc}}
            historical_returns: Dictionary of historical returns by symbol
            correlation_matrix: Optional correlation matrix between assets
            
        Returns:
            RiskMetrics object with portfolio risk calculations
        """
        # Placeholder implementation - would need more complex calculations in practice
        weights = np.array([pos['weight'] for pos in positions.values()])
        portfolio_returns = np.zeros(len(next(iter(historical_returns.values()))))
        
        # Calculate portfolio returns
        for symbol, pos in positions.items():
            if symbol in historical_returns:
                portfolio_returns += pos['weight'] * historical_returns[symbol]
        
        # Calculate risk metrics
        var_95 = self.calculate_var(portfolio_returns, 0.95)
        var_99 = self.calculate_var(portfolio_returns, 0.99)
        es = self.calculate_expected_shortfall(portfolio_returns, 0.95)
        
        # Calculate max drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # Calculate volatility
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualize
        
        # Calculate Sharpe ratio if we have return data
        sharpe_ratio = None
        if len(portfolio_returns) > 0:
            avg_return = portfolio_returns.mean() * 252  # Annualize
            risk_free_rate = 0.02  # Placeholder - should use actual risk-free rate
            sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        return RiskMetrics(
            value_at_risk_95=var_95,
            value_at_risk_99=var_99,
            expected_shortfall=es,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio
        )
    
    def validate_trade(self, 
                       trade_symbol: str,
                       trade_size: float,
                       trade_direction: str,  # "BUY" or "SELL"
                       current_portfolio: Dict[str, Dict],
                       historical_data: Dict[str, np.ndarray],
                       sector_data: Dict[str, str]) -> Tuple[bool, str]:
        """
        Validate a trade against risk management rules
        
        Args:
            trade_symbol: Symbol to trade
            trade_size: Size of the trade in currency amount
            trade_direction: "BUY" or "SELL"
            current_portfolio: Current portfolio holdings
            historical_data: Historical price data
            sector_data: Sector classification data
            
        Returns:
            Tuple of (is_valid, reason)
        """
        portfolio_value = sum(pos.get('value', 0) for pos in current_portfolio.values())
        
        # Check if trade size exceeds max position size
        if trade_size / portfolio_value > self.max_position_size:
            return False, f"Trade exceeds maximum position size of {self.max_position_size*100}% of portfolio"
        
        # Check sector exposure
        if trade_symbol in sector_data:
            sector = sector_data[trade_symbol]
            current_sector_exposure = sum(
                pos.get('value', 0) for sym, pos in current_portfolio.items() 
                if sym in sector_data and sector_data[sym] == sector
            ) / portfolio_value
            
            new_sector_exposure = current_sector_exposure
            if trade_direction == "BUY":
                new_sector_exposure += trade_size / portfolio_value
                
            if new_sector_exposure > self.max_sector_exposure:
                return False, f"Trade would exceed maximum sector exposure of {self.max_sector_exposure*100}% for {sector}"
        
        # More sophisticated checks could be added here:
        # - Correlation check
        # - VaR contribution
        # - Liquidity check
        
        return True, "Trade passes risk management checks"
