import json
import logging
import datetime
import os
from typing import Dict, Any, Optional

class AuditTrail:
    """Records all significant actions for compliance and debugging purposes."""
    
    def __init__(self, log_dir: str = "./logs/audit"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("audit_trail")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(log_dir, f"audit_{datetime.datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
    
    def record_action(self, 
                     action_type: str, 
                     details: Dict[str, Any], 
                     user_id: Optional[str] = None,
                     source_ip: Optional[str] = None) -> None:
        """
        Record an action in the audit trail.
        
        Args:
            action_type: Type of action (e.g., "TRADE_EXECUTED", "LOGIN", "STRATEGY_CHANGE")
            details: Dictionary with action details
            user_id: ID of the user who performed the action (if applicable)
            source_ip: IP address where the action originated (if applicable)
        """
        audit_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action_type,
            "details": details
        }
        
        if user_id:
            audit_entry["user_id"] = user_id
        
        if source_ip:
            audit_entry["source_ip"] = source_ip
            
        self.logger.info(json.dumps(audit_entry))
        
    def query_actions(self, 
                      start_time: datetime.datetime,
                      end_time: datetime.datetime,
                      action_type: Optional[str] = None) -> list:
        """
        Query the audit trail for actions within a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            action_type: Optional filter for specific action types
            
        Returns:
            List of matching audit entries
        """
        # This is a simple implementation that could be enhanced with a proper database
        results = []
        
        # Determine which log files to search based on date range
        start_date = start_time.date()
        end_date = end_time.date()
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            log_file = os.path.join(self.log_dir, f"audit_{date_str}.log")
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    for line in f:
                        if action_type and action_type not in line:
                            continue
                        
                        # Parse the log entry and check timestamp
                        try:
                            # Extract JSON part from log line
                            json_str = line.split(' - INFO - ')[1]
                            entry = json.loads(json_str)
                            
                            entry_time = datetime.datetime.fromisoformat(entry["timestamp"])
                            if start_time <= entry_time <= end_time:
                                results.append(entry)
                        except (IndexError, json.JSONDecodeError, KeyError):
                            # Skip malformed entries
                            continue
            
            current_date += datetime.timedelta(days=1)
            
        return results
