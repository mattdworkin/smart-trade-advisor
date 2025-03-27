@echo off
call venv\Scripts\activate.bat

REM Try an older version of foil
pip install foil==0.2.6

echo Done!
pause 