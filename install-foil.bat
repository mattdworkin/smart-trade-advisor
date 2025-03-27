@echo off
call venv\Scripts\activate.bat

REM Try installing foil from source
pip install --no-binary=foil foil

echo Done!
pause 