@echo off
echo ===================================
echo Fixing XGBoost Method Error
echo ===================================
echo.

python fix_xgboost_method.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===================================
    echo Fix completed successfully!
    echo.
    echo Now you can run the lightweight training script:
    echo train_model_light.bat AAPL,MSFT random_forest
    echo.
    echo Or try XGBoost with reduced resource usage:
    echo train_model_light.bat AAPL xgboost
    echo ===================================
) else (
    echo.
    echo ===================================
    echo Fix encountered some errors.
    echo Please check the error messages above.
    echo ===================================
)

echo.
echo Press any key to exit...
pause > nul 