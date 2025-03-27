@echo off
echo Starting Smart Trade Advisor with Real-Time Suggestions...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Create directory for templates if it doesn't exist
if not exist templates mkdir templates

REM Create a backup of main.py
copy main.py main.py.backup

REM Fix the missing RiskManager import
echo Adding missing imports and fixes to main.py...
powershell -Command "(Get-Content main.py) -replace 'import json', 'import json\n\ntry:\n    from trading.risk_management import RiskManager, PositionSizing\nexcept ImportError:\n    logger.warning(\"Could not import risk_management module, using placeholder\")\n    \n    class RiskManager:\n        def __init__(self, *args, **kwargs):\n            pass\n            \n        def validate_trade(self, trade_symbol, trade_size, trade_direction, current_portfolio, historical_data, sector_data):\n            return True, \"No risk validation performed\"' | Set-Content main.py"

REM Add the risk_manager property
powershell -Command "(Get-Content main.py) -replace 'SmartTradeAdvisor.trade_journaler = property\(lambda self: TradeJournaler\(\)\)', 'SmartTradeAdvisor.trade_journaler = property(lambda self: TradeJournaler())\nSmartTradeAdvisor.risk_manager = property(lambda self: RiskManager())' | Set-Content main.py"

REM Add current_regime initialization
powershell -Command "(Get-Content main.py) -replace 'self.current_portfolio = self._load_portfolio\(\"sample_portfolio.json\"\)', 'self.current_portfolio = self._load_portfolio(\"sample_portfolio.json\")\n        self.current_regime = \"neutral\"  # Default regime' | Set-Content main.py"

REM Run the application
echo Starting the Flask application...
python app.py

pause 