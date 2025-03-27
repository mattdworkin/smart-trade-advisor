@echo off
echo Creating virtual environment if it doesn't exist...
if not exist venv (
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo Starting Smart Trade Advisor...
python launch.py --web 