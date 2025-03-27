@echo off
echo Starting Smart Trade Advisor Web Interface...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the application
python app.py

pause 