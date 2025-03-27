@echo off
echo Installing all Smart Trade Advisor features...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install machine learning packages
echo Installing machine learning packages...
pip install scikit-learn xgboost matplotlib joblib

REM Install day trading packages
echo Installing day trading packages...
pip install yfinance pandas-ta

REM Install scanning and news packages
echo Installing scanning and news packages...
pip install requests beautifulsoup4 seaborn

echo All packages installed!
echo.
echo Smart Trade Advisor is now ready with all features:
echo - Machine learning predictions
echo - Real-time day trading
echo - Scanner presets
echo - Sector analysis
echo - News integration
echo - Intraday alert system
echo.
echo Start using the full system with:
echo python main.py
echo.
pause 