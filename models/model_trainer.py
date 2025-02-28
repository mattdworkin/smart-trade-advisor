from backtesting.backtest import Backtester, DataHandler

def train_model():
    # [Existing training code]
    
    # After training completes
    if config['run_backtest']:
        data_handler = DataHandler(start_date='2010-01-01', end_date='2023-12-31')
        backtester = Backtester(trained_model)
        metrics = backtester.run_walk_forward_test(data_handler)
        print(f"\nBacktest Metrics:\n{metrics}")
        metrics.to_csv('backtest_metrics.csv')