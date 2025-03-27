@echo off
echo Installing remaining dependencies...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install additional packages
echo Installing matplotlib and other visualization libraries...
pip install matplotlib seaborn plotly

echo Installing technical analysis libraries...
pip install ta yfinance alphavantage

echo Verifying installations...
pip list

echo Done! Your system should now be ready to run.
pause 