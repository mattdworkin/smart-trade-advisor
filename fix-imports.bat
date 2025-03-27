@echo off
echo Fixing import issues...

REM Create backup of metrics.py first
copy backtesting\metrics.py backtesting\metrics.py.bak

REM Fix the metrics.py file
echo import numpy as np > backtesting\metrics.py
echo import pandas as pd >> backtesting\metrics.py
echo from typing import Dict, List, Tuple, Any, Optional >> backtesting\metrics.py
echo. >> backtesting\metrics.py
type backtesting\metrics.py.bak | findstr /v "import" >> backtesting\metrics.py

echo Fixing import issues in main.py...

REM Create a backup of main.py
copy main.py main.py.bak

REM Replace the incorrect import statement
powershell -Command "(Get-Content main.py) -replace 'from backtesting.backtest import Backtester', 'from backtesting.backtest import Backtest' | Set-Content main.py"
powershell -Command "(Get-Content main.py) -replace 'from backtesting.backtest import Backtester as Backtest', 'from backtesting.backtest import Backtest' | Set-Content main.py"

echo Fixing import issues in model_trainer.py...

REM Create a backup of model_trainer.py
copy models\model_trainer.py models\model_trainer.py.bak

REM Replace the incorrect import statement
powershell -Command "(Get-Content models\model_trainer.py) -replace 'from backtesting.backtest import Backtester', 'from backtesting.backtest import Backtest' | Set-Content models\model_trainer.py"

echo Fixes applied! Try running your application now.
pause 