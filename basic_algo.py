import yfinance as yf
import pandas as pd

def get_stock_data(ticker, period='1mo', interval='1d'):
    """
    Fetch historical stock data.
    """
    data = yf.download(ticker, period=period, interval=interval)
    return data

def calculate_sma(data, window=5):
    """
    Calculate the Simple Moving Average (SMA).
    """
    data['SMA'] = data['Close'].rolling(window=window).mean()
    return data

def generate_trade_signal(data):
    """
    Generate a trade signal based on the SMA strategy.
    """
    # Ensure we have enough data for SMA calculation
    if len(data) < 5:
        return "Insufficient data"
    
    latest_price = data['Close'].iloc[-1]
    latest_sma = data['SMA'].iloc[-1]
    
    if latest_price > latest_sma:
        return "BUY"
    elif latest_price < latest_sma:
        return "SELL"
    else:
        return "HOLD"

if __name__ == '__main__':
    ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    data = get_stock_data(ticker)
    if data.empty:
        print("No data found for ticker:", ticker)
    else:
        data = calculate_sma(data, window=5)
        signal = generate_trade_signal(data)
        print(f"Trade signal for {ticker}: {signal}")
