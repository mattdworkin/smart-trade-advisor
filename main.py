# main.py updates
from backtesting.backtest import Backtester
from models.model_trainer import train_model

def main():
    # Load user positions
    positions = load_positions()
    
    # Train and validate
    model = train_model()
    backtester = Backtester(model)
    metrics = backtester.run_walk_forward_test()
    
    if metrics['Sharpe Ratio'].values[0] < 1.5:
        raise Exception("Strategy failed backtest - aborting live trading")
    
    # Initiate live trading
    trading_engine = LiveTrading(model, positions)
    trading_engine.run()