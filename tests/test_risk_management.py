import unittest
import numpy as np
from trading.risk_management import RiskManager, PositionSizing

class TestRiskManagement(unittest.TestCase):
    def setUp(self):
        self.risk_manager = RiskManager()
        self.position_sizer = PositionSizing()
    
    def test_kelly_criterion(self):
        # Test with 60% win rate and 2:1 win/loss ratio
        size = self.position_sizer.kelly_criterion(win_rate=0.6, win_loss_ratio=2.0)
        self.assertGreaterEqual(size, 0)
        self.assertLessEqual(size, 1)
    
    def test_percent_risk_sizing(self):
        capital = 100000
        max_risk = 0.01
        entry = 100
        stop = 95
        
        size = self.position_sizer.percent_risk_sizing(
            capital=capital,
            max_risk_percent=max_risk,
            entry_price=entry,
            stop_price=stop
        )
        
        expected = (capital * max_risk) / (entry - stop)
        self.assertEqual(size, expected)
    
    def test_calculate_var(self):
        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, 1000)
        
        # Calculate 95% VaR
        var_95 = self.risk_manager.calculate_var(returns, confidence_level=0.95)
        
        # VaR should be positive (representing a loss)
        self.assertGreater(var_95, 0)
        # 99% VaR should be greater than 95% VaR
        var_99 = self.risk_manager.calculate_var(returns, confidence_level=0.99)
        self.assertGreater(var_99, var_95)

if __name__ == '__main__':
    unittest.main() 