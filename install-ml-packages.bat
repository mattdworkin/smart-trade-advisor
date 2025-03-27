@echo off
echo Installing machine learning packages for Smart Trade Advisor...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install required packages
echo Installing scikit-learn...
pip install scikit-learn

echo Installing XGBoost...
pip install xgboost

echo Installing matplotlib...
pip install matplotlib

echo Installing joblib...
pip install joblib

echo Installing yfinance (for data download)...
pip install yfinance

echo All packages installed!
echo.
echo Now you can train real machine learning models with:
echo python -m models.model_trainer
echo.
pause 