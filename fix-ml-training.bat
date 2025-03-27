@echo off
echo ===================================
echo Smart Trade Advisor - ML Training Fix
echo ===================================
echo.
echo This script will fix common issues with ML model training:
echo 1. Check and install required dependencies
echo 2. Optimize model_trainer.py to use less resources
echo 3. Create a lightweight training alternative
echo.
echo Press any key to continue...
pause > nul

python fix_ml_training.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===================================
    echo Fix completed successfully!
    echo.
    echo You can now run model training in two ways:
    echo 1. Through the application menu (option 10)
    echo 2. Using the new lightweight script:
    echo    train_model_light.bat AAPL,MSFT,GOOGL xgboost
    echo.
    echo The lightweight option uses fewer resources and
    echo should resolve CPU/disk issues on your laptop.
    echo ===================================
) else (
    echo.
    echo ===================================
    echo Fix encountered some errors.
    echo Please check the log output above.
    echo.
    echo You may need to manually install dependencies:
    echo pip install numpy pandas scikit-learn xgboost==1.7.3 matplotlib joblib yfinance
    echo ===================================
)

echo.
echo Press any key to exit...
pause > nul 