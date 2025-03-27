@echo off
echo Running Smart Trade Advisor with minimal dependencies...

REM Create a fresh virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Make sure to activate the virtual environment
call venv\Scripts\activate.bat

REM Show which Python we're using (for debugging)
echo Using Python from:
where python

REM Show pip version and location
echo Pip version:
pip --version

REM Force upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install packages with verbose output
echo Installing Flask (verbose)...
python -m pip install flask --verbose

echo Installing additional packages...
python -m pip install pandas numpy

REM Verify Flask is installed
echo Checking if Flask is installed:
pip list | findstr Flask

REM Run the application
echo Starting the application...
python app.py

pause