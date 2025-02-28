# trading/regime_detector.py
from changefinder import ChangeFinder

class MarketRegimeDetector:
    def __init__(self):
        self.cf = ChangeFinder(r=0.01, order=1, smooth=7)
        
    def detect_regime(self, price_series):
        anomaly_score = []
        for p in price_series:
            anomaly_score.append(self.cf.update(p))
            
        if np.mean(anomaly_score[-10:]) > 2.0:
            return "high_volatility"
        return "normal"
        
# Integrated into main loop:
regime = detector.detect_regime(prices)
if regime == "high_volatility":
    trading_engine.pause_trading()