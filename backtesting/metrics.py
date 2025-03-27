import numpy as np 
import pandas as pd 
from typing import Dict, List, Tuple, Any, Optional 
 
 

def calculate_metrics(results: Dict) -> pd.DataFrame:
    returns = results['returns']
    drawdown = results['drawdown']
    
    metrics = {
        'Sharpe Ratio': _annualized_sharpe(returns),
        'Sortino Ratio': _sortino_ratio(returns),
        'Max Drawdown': drawdown.min(),
        'Win Rate': (returns > 0).mean(),
        'Profit Factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()),
        'Kelly Criterion': _kelly_criterion(returns)
    }
    
    _plot_equity_curve(returns)
    return pd.DataFrame(metrics, index=[0])

def _annualized_sharpe(returns: pd.Series, risk_free=0.0) -> float:
    excess_returns = returns - risk_free
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def _sortino_ratio(returns: pd.Series, risk_free=0.0) -> float:
    downside_returns = returns[returns < 0]
    return (returns.mean() - risk_free) / downside_returns.std()

def _kelly_criterion(returns: pd.Series) -> float:
    win_prob = (returns > 0).mean()
    win_loss_ratio = returns[returns > 0].mean() / abs(returns[returns < 0].mean())
    return win_prob - (1 - win_prob)/win_loss_ratio

def _plot_equity_curve(returns: pd.Series):
    plt.figure(figsize=(10,6))
    (1 + returns).cumprod().plot(title='Strategy Equity Curve')
    plt.savefig('backtest_results.png')
    plt.close()
