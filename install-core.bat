@echo off
echo Installing core dependencies...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install matplotlib and visualization packages
echo Installing matplotlib and visualization libraries...
pip install matplotlib seaborn plotly

REM Install core trading and data packages (skip foil)
echo Installing technical analysis libraries...
pip install ta yfinance

echo Installation complete. Your system should now be ready to run.
pause 