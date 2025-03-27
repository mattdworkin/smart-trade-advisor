@echo off
echo Installing day trading packages for Smart Trade Advisor...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install required packages
echo Installing yfinance for market data...
pip install yfinance

echo Installing pandas-ta for technical indicators...
pip install pandas-ta

echo All packages installed!
echo.
echo Now you can use the day trading features with:
echo python main.py
echo.
pause 