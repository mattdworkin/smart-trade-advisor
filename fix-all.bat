@echo off
echo Fixing syntax errors in your files...

REM Make backup copies
copy main.py main.py.broken
copy trading\regime_detector.py trading\regime_detector.py.broken

REM Get the text of the fixed files from another file
echo Writing fixed main.py...
copy main-fixed.py main.py

echo Writing fixed regime_detector.py...
copy regime_detector-fixed.py trading\regime_detector.py

echo Files have been fixed! You can now run your application.
echo.
echo To run the web interface:
echo .\run-web.bat
echo.
pause 