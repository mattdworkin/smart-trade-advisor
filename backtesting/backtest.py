import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, Tuple
from .metrics import calculate_metrics

class DataHandler:
    def __init__(self, start_date: str, end_date: str):
        self.start = pd.to_datetime(start_date)
        self.end = pd.to_datetime(end_date)
        
    def get_historical_data(self, ticker: str) -> pd.DataFrame:
        """Fetch OHLCV data with technical indicators"""
        df = yf.download(ticker, start=self.start, end=self.end)
        df = self._add_technical_indicators(df)
        return df

class Backtester:
    def __init__(self, model, initial_capital: float = 100000):
        self.model = model
        self.initial_capital = initial_capital
        self.current_balance = initial_capital
        self.positions = {}
        self.trade_history = []
        
    def run_walk_forward_test(self, data_handler: DataHandler,
                            train_years: int = 3, test_months: int = 3):
        """Walk-forward testing with expanding window"""
        current_date = data_handler.start + pd.DateOffset(years=train_years)
        
        while current_date <= data_handler.end:
            train_data = data_handler.get_historical_data(
                start=data_handler.start,
                end=current_date - pd.DateOffset(months=test_months)
            )
            test_data = data_handler.get_historical_data(
                start=current_date - pd.DateOffset(months=test_months),
                end=current_date
            )
            
            self.model.train(train_data)
            results = self._test_on_period(test_data)
            
            current_date += pd.DateOffset(months=test_months)
            
        return calculate_metrics(results)

    def _test_on_period(self, data: pd.DataFrame) -> Dict:
        """Simulate trading on unseen data"""
        portfolio_values = []
        for idx, row in data.iterrows():
            prediction = self.model.predict(row)
            self._execute_trades(prediction, row)
            
            # Update portfolio value
            total = self.current_balance
            for ticker, position in self.positions.items():
                total += position['shares'] * row['Close']
            portfolio_values.append(total)
            
        return {
            'returns': pd.Series(portfolio_values).pct_change(),
            'drawdown': self._calculate_drawdown(portfolio_values)
        }