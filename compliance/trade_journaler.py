import csv
import os
import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class TradeRecord:
    """Data class to store trade information"""
    timestamp: datetime.datetime
    symbol: str
    action: str  # "BUY" or "SELL"
    quantity: float
    price: float
    total_value: float
    strategy_name: str
    trade_id: str
    portfolio_id: str
    execution_details: Dict[str, Any] = None
    notes: str = ""
    
    @property
    def trade_date(self) -> datetime.date:
        return self.timestamp.date()

class TradeJournaler:
    """
    Records all trades for analysis, reporting, and compliance purposes.
    Maintains a detailed journal of all trading activity.
    """
    
    def __init__(self, journal_dir: str = "./data/trade_journal"):
        self.journal_dir = journal_dir
        os.makedirs(journal_dir, exist_ok=True)
        
        # Ensure we have the master index file
        self.master_index_path = os.path.join(journal_dir, "master_index.csv")
        if not os.path.exists(self.master_index_path):
            with open(self.master_index_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["trade_id", "date", "symbol", "action", "file_path"])
    
    def record_trade(self, trade: TradeRecord) -> None:
        """
        Record a trade in the journal
        
        Args:
            trade: The TradeRecord object containing trade details
        """
        # Create date directory if it doesn't exist
        date_str = trade.trade_date.strftime("%Y-%m-%d")
        date_dir = os.path.join(self.journal_dir, date_str)
        os.makedirs(date_dir, exist_ok=True)
        
        # Create the trade file
        trade_file = os.path.join(date_dir, f"{trade.trade_id}.csv")
        trade_exists = os.path.exists(trade_file)
        
        # Write trade details
        with open(trade_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Field", "Value"])
            writer.writerow(["trade_id", trade.trade_id])
            writer.writerow(["timestamp", trade.timestamp.isoformat()])
            writer.writerow(["symbol", trade.symbol])
            writer.writerow(["action", trade.action])
            writer.writerow(["quantity", trade.quantity])
            writer.writerow(["price", trade.price])
            writer.writerow(["total_value", trade.total_value])
            writer.writerow(["strategy_name", trade.strategy_name])
            writer.writerow(["portfolio_id", trade.portfolio_id])
            
            # Add execution details
            if trade.execution_details:
                for key, value in trade.execution_details.items():
                    writer.writerow([f"execution_{key}", value])
            
            # Add notes if any
            if trade.notes:
                writer.writerow(["notes", trade.notes])
        
        # Update the master index if this is a new trade
        if not trade_exists:
            with open(self.master_index_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade.trade_id,
                    date_str,
                    trade.symbol,
                    trade.action,
                    os.path.relpath(trade_file, self.journal_dir)
                ])
    
    def get_trades_by_date(self, start_date: datetime.date, 
                          end_date: Optional[datetime.date] = None) -> List[TradeRecord]:
        """
        Retrieve trades within a date range
        
        Args:
            start_date: Beginning date to retrieve trades from
            end_date: End date, defaults to start_date if not provided
            
        Returns:
            List of TradeRecord objects within the date range
        """
        if end_date is None:
            end_date = start_date
            
        trades = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            date_dir = os.path.join(self.journal_dir, date_str)
            
            if os.path.exists(date_dir):
                for filename in os.listdir(date_dir):
                    if filename.endswith(".csv"):
                        trade_file = os.path.join(date_dir, filename)
                        trade = self._parse_trade_file(trade_file)
                        if trade:
                            trades.append(trade)
            
            current_date += datetime.timedelta(days=1)
            
        return trades
    
    def get_trade_by_id(self, trade_id: str) -> Optional[TradeRecord]:
        """
        Retrieve a specific trade by its ID
        
        Args:
            trade_id: The unique ID of the trade
            
        Returns:
            TradeRecord if found, None otherwise
        """
        # Search in master index
        with open(self.master_index_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            for row in reader:
                if row[0] == trade_id:
                    trade_file = os.path.join(self.journal_dir, row[4])
                    return self._parse_trade_file(trade_file)
        
        return None
    
    def _parse_trade_file(self, file_path: str) -> Optional[TradeRecord]:
        """Parse a trade file into a TradeRecord object"""
        if not os.path.exists(file_path):
            return None
            
        trade_data = {}
        execution_details = {}
        
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            for row in reader:
                if len(row) < 2:
                    continue
                    
                field, value = row[0], row[1]
                
                if field.startswith("execution_"):
                    execution_details[field[10:]] = value
                else:
                    trade_data[field] = value
        
        try:
            trade = TradeRecord(
                timestamp=datetime.datetime.fromisoformat(trade_data.get("timestamp")),
                symbol=trade_data.get("symbol"),
                action=trade_data.get("action"),
                quantity=float(trade_data.get("quantity", 0)),
                price=float(trade_data.get("price", 0)),
                total_value=float(trade_data.get("total_value", 0)),
                strategy_name=trade_data.get("strategy_name", ""),
                trade_id=trade_data.get("trade_id"),
                portfolio_id=trade_data.get("portfolio_id", ""),
                execution_details=execution_details,
                notes=trade_data.get("notes", "")
            )
            return trade
        except (ValueError, KeyError):
            # If we can't parse the trade file properly, return None
            return None
