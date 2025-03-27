# fix_ml_training.py - Script to fix common ML training issues
import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ml_training_fix")

def check_and_install_dependencies():
    """Check for required packages and install them if missing"""
    required_packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost==1.7.3",  # Specify version for compatibility
        "matplotlib",
        "joblib",
        "yfinance"
    ]
    
    logger.info("Checking ML dependencies...")
    for package in required_packages:
        try:
            package_name = package.split("==")[0]
            __import__(package_name)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            logger.warning(f"✗ {package} is not installed. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"✓ {package} installed successfully")
            except Exception as e:
                logger.error(f"Failed to install {package}: {e}")
                return False
    
    return True

def create_required_directories():
    """Create necessary directories for the ML pipeline"""
    directories = [
        "data/processed",
        "models/trained",
        "results/figures",
    ]
    
    logger.info("Creating required directories...")
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✓ Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            return False
    
    return True

def optimize_training_config():
    """Create an optimized training configuration file"""
    optimized_config = {
        "training": {
            "default_model_type": "xgboost",
            "default_symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
            "prediction_target": "next_day_return",
            "prediction_horizon": 1,
            "default_features": ["rsi", "macd", "bollinger", "volume", "momentum"],
            "lookback_period": 20,
            "test_size": 0.2,
            "cv_folds": 2,
            "run_backtest": False,
            "resource_optimization": {
                "threads": 2,  # Limit threads to reduce CPU usage
                "early_stopping_rounds": 10,
                "max_depth": 5
            }
        }
    }
    
    import json
    try:
        with open("ml_config.json", "w") as f:
            json.dump(optimized_config, f, indent=4)
        logger.info("✓ Created optimized training configuration")
        return True
    except Exception as e:
        logger.error(f"Failed to create configuration file: {e}")
        return False

def patch_model_trainer():
    """Patch the ModelTrainer class to be more resource-efficient"""
    patch_code = """
# XGBoost resource optimization patch
def _create_xgboost(self):
    import xgboost as xgb
    return xgb.XGBRegressor(
        n_estimators=50,  # Reduced from 100
        learning_rate=0.1,
        max_depth=3,     # Reduced from 5
        random_state=42,
        n_jobs=2,        # Limit parallel jobs
        subsample=0.8,   # Use 80% of data for training
        colsample_bytree=0.8  # Use 80% of features
    )
"""
    
    try:
        # Add the optimized function to the model_trainer.py file
        with open("models/model_trainer.py", "r") as f:
            content = f.read()
        
        # Check if the file already contains our optimized version
        if "subsample=0.8" not in content:
            # Find the original function
            original_func = "    def _create_xgboost(self):"
            
            if original_func in content:
                # Find the end of the function
                start_idx = content.find(original_func)
                next_def_idx = content.find("    def ", start_idx + len(original_func))
                
                # Replace the original function with our optimized version
                modified_content = content[:start_idx] + patch_code + content[next_def_idx:]
                
                with open("models/model_trainer.py", "w") as f:
                    f.write(modified_content)
                
                logger.info("✓ Patched model_trainer.py with resource optimization")
            else:
                logger.warning("Could not find _create_xgboost function in model_trainer.py")
        else:
            logger.info("✓ model_trainer.py already optimized")
        
        return True
    except Exception as e:
        logger.error(f"Failed to patch model_trainer.py: {e}")
        return False

def create_lightweight_train_function():
    """Create a lightweight training script"""
    with open("train_model_light.py", "w") as f:
        f.write("""
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
""")
    logger.info("✓ Created lightweight training script: train_model_light.py")
    return True

def create_training_batch_script():
    """Create a batch script to run the lightweight training"""
    with open("train_model_light.bat", "w") as f:
        f.write("""@echo off
echo Running lightweight model training...
python train_model_light.py %*
if %ERRORLEVEL% EQU 0 (
    echo Training completed successfully!
) else (
    echo Training failed with error code %ERRORLEVEL%
)
pause
""")
    logger.info("✓ Created training batch script: train_model_light.bat")
    return True

def patch_main_py_training():
    """Patch the train_models method in main.py to handle errors better"""
    try:
        with open("main.py", "r") as f:
            content = f.read()
            
        # Find the train_models method
        start_marker = "def train_models(self, symbols=None, model_type='xgboost') -> Dict:"
        if start_marker in content:
            # Find position of the method
            start_idx = content.find(start_marker)
            
            # Find where the try block starts
            try_marker = "        try:"
            try_pos = content.find(try_marker, start_idx)
            
            if try_pos > -1:
                # Add error-handling code before the try block
                insert_code = """        # Set resource limits for training
        if model_type == 'xgboost':
            try:
                import xgboost as xgb
                # Check if we can load xgboost with no errors
            except ImportError as e:
                logger.error(f"XGBoost import error: {e}")
                logger.info("Falling back to random_forest model")
                model_type = 'random_forest'
            except Exception as e:
                logger.error(f"XGBoost error: {e}")
                logger.info("Falling back to random_forest model")
                model_type = 'random_forest'
                
        # Limit resources for lighter training
        resource_limits = {
            'n_estimators': 50,  # Fewer trees
            'threads': 2,        # Limit CPU usage
            'subsample': 0.8     # Use subset of data
        }
                
"""
                modified_content = content[:try_pos] + insert_code + content[try_pos:]
                
                # Write back to file
                with open("main.py", "w") as f:
                    f.write(modified_content)
                    
                logger.info("✓ Patched main.py train_models method with better error handling")
                return True
            else:
                logger.warning("Could not find try block in train_models method")
        else:
            logger.warning("Could not find train_models method in main.py")
        
        return False
    except Exception as e:
        logger.error(f"Failed to patch main.py: {e}")
        return False

def main():
    """Main function to fix ML training issues"""
    logger.info("Starting ML training fix script...")
    
    # Check and install dependencies
    if not check_and_install_dependencies():
        logger.error("Failed to install dependencies. Please install manually.")
        return False
    
    # Create required directories
    if not create_required_directories():
        logger.error("Failed to create required directories.")
        return False
    
    # Create optimized training config
    if not optimize_training_config():
        logger.warning("Failed to create optimized config. Continuing...")
    
    # Patch model_trainer.py
    if not patch_model_trainer():
        logger.warning("Failed to patch model_trainer.py. Continuing...")
    
    # Create lightweight training script
    if not create_lightweight_train_function():
        logger.warning("Failed to create lightweight training script. Continuing...")
    
    # Create batch script
    if not create_training_batch_script():
        logger.warning("Failed to create batch script. Continuing...")
    
    # Patch main.py
    if not patch_main_py_training():
        logger.warning("Failed to patch main.py. Continuing...")
    
    logger.info("ML training fix completed!")
    logger.info("You can now run 'train_model_light.bat' for more efficient training.")
    return True

if __name__ == "__main__":
    main() 