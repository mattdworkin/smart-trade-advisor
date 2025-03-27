from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import datetime
from main import SmartTradeAdvisor

app = Flask(__name__)

# Initialize the advisor
advisor = SmartTradeAdvisor()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_system():
    """Start the trading system"""
    advisor.start()
    return jsonify({"status": "success", "message": "System started successfully"})

@app.route('/stop', methods=['POST'])
def stop_system():
    """Stop the trading system"""
    advisor.stop()
    return jsonify({"status": "success", "message": "System stopped successfully"})

@app.route('/analysis', methods=['POST'])
def run_analysis():
    """Run analysis and get trade suggestions"""
    suggestions = advisor.run_analysis_cycle()
    return jsonify({
        "status": "success", 
        "count": len(suggestions),
        "suggestions": suggestions
    })

@app.route('/execute_trade', methods=['POST'])
def execute_trade():
    """Execute a trade"""
    data = request.json
    trade = {
        "symbol": data.get("symbol"),
        "action": data.get("action"),
        "quantity": float(data.get("quantity", 0)),
        "strategy": data.get("strategy", "Manual"),
        "reason": data.get("reason", "User initiated")
    }
    
    result = advisor.execute_trade(trade)
    return jsonify({
        "status": "success",
        "trade_id": result.get("trade_id"),
        "executed_price": result.get("executed_price")
    })

@app.route('/portfolio')
def get_portfolio():
    """Get current portfolio data"""
    portfolio = advisor.current_portfolio
    
    # Calculate current values
    positions_with_values = []
    for symbol, pos in portfolio.get('positions', {}).items():
        current_price = advisor.realtime_stream.get_last_price(symbol) or pos.get('average_price', 0)
        current_value = pos.get('shares', 0) * current_price
        cost_basis = pos.get('shares', 0) * pos.get('average_price', 0)
        profit_loss = current_value - cost_basis
        profit_loss_pct = 100 * profit_loss / cost_basis if cost_basis > 0 else 0
        
        positions_with_values.append({
            "symbol": symbol,
            "shares": pos.get('shares', 0),
            "average_price": pos.get('average_price', 0),
            "current_price": current_price,
            "current_value": current_value,
            "profit_loss": profit_loss,
            "profit_loss_pct": profit_loss_pct
        })
    
    return jsonify({
        "cash": portfolio.get('cash', 0),
        "positions": positions_with_values
    })

@app.route('/trades')
def get_trades():
    """Get recent trades"""
    days = int(request.args.get('days', 7))
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days)
    
    trades = advisor.trade_journaler.get_trades_by_date(
        start_date=start_date,
        end_date=today
    )
    
    trade_data = []
    for trade in trades:
        trade_data.append({
            "trade_id": trade.trade_id,
            "timestamp": trade.timestamp.isoformat(),
            "symbol": trade.symbol,
            "action": trade.action,
            "quantity": trade.quantity,
            "price": trade.price,
            "total_value": trade.total_value,
            "strategy": trade.strategy_name,
            "notes": trade.notes
        })
    
    return jsonify(trade_data)

if __name__ == '__main__':
    app.run(debug=True) 