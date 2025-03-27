@echo off
echo Starting Smart Trade Advisor...

REM Create necessary directories
if not exist templates mkdir templates
if not exist static\css mkdir static\css
if not exist static\js mkdir static\js
if not exist logs mkdir logs
if not exist data\cache mkdir data\cache

REM Clean up the old environment if needed
if exist venv (
    echo Removing old virtual environment...
    rmdir /s /q venv
)

REM Create a fresh virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install setuptools wheel

REM Try the compatible requirements first
echo Installing essential dependencies...
pip install -r requirements-compatible.txt

REM Run the application
echo Starting the application...
python app.py 