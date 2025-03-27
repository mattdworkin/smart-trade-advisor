# Create necessary directories
New-Item -ItemType Directory -Path logs -Force
New-Item -ItemType Directory -Path data\cache -Force

# Check for virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
} else {
    .\venv\Scripts\Activate.ps1
}

# Run the application
Write-Host "Starting Smart Trade Advisor..."
python app.py 