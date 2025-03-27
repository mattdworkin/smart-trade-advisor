@echo off
echo Running lightweight model training...
python train_model_light.py %*
if %ERRORLEVEL% EQU 0 (
    echo Training completed successfully!
) else (
    echo Training failed with error code %ERRORLEVEL%
)
pause
