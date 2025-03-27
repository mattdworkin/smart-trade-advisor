#!/bin/bash

# Create necessary directories
mkdir -p logs data/cache

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run the application
echo "Starting Smart Trade Advisor..."
python app.py 