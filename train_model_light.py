
# train_model_light.py - Lightweight model training script
from models.model_trainer import train_model
import sys
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def train_lightweight():
    # Load optimized config if available
    try:
        with open("ml_config.json", "r") as f:
            ml_config = json.load(f)
            training_config = ml_config.get("training", {})
    except:
        training_config = {}
        
    # Get arguments
    if len(sys.argv) > 1:
        symbols = sys.argv[1].split(',')
    else:
        symbols = training_config.get("default_symbols", ["AAPL", "MSFT"])
    
    if len(sys.argv) > 2:
        model_type = sys.argv[2]
    else:
        model_type = training_config.get("default_model_type", "random_forest")
    
    # Create lightweight config    
    config = {
        'model_type': model_type,
        'symbols': symbols,
        'prediction_target': training_config.get("prediction_target", "next_day_return"),
        'prediction_horizon': training_config.get("prediction_horizon", 1),
        'features': training_config.get("default_features", ["rsi", "macd", "bollinger"]),
        'lookback_period': training_config.get("lookback_period", 20),
        'test_size': training_config.get("test_size", 0.2),
        'cv_folds': training_config.get("cv_folds", 2),
        'run_backtest': training_config.get("run_backtest", False)
    }
    
    # Add resource optimization if available
    resource_opt = training_config.get("resource_optimization", {})
    if model_type == "xgboost" and resource_opt:
        config['resource_optimization'] = resource_opt
    
    logger.info(f"Training with config: {config}")
    
    # Train model
    result = train_model(config)
    
    # Print results
    if result:
        for symbol, perf in result.get('performance', {}).items():
            logger.info(f"{symbol} Performance:")
            logger.info(f"  Direction Accuracy: {perf.get('direction_accuracy', 0):.2%}")
            logger.info(f"  Profit Factor: {perf.get('profit_factor', 0):.2f}")
    else:
        logger.error("Training failed")

if __name__ == "__main__":
    train_lightweight()
