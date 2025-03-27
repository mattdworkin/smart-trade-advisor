import re

# Read the model_trainer.py file
with open('models/model_trainer.py', 'r') as f:
    content = f.read()

# Check if the standalone _create_xgboost function exists
standalone_pattern = r'\ndef _create_xgboost\(self\):'
match = re.search(standalone_pattern, content)

if match:
    print("Found incorrectly indented _create_xgboost method. Fixing...")
    
    # Remove the standalone function
    modified_content = re.sub(r'\n# XGBoost resource optimization patch\ndef _create_xgboost\(self\):.*?colsample_bytree=0\.8  # Use 80% of features\n    \)', '', content, flags=re.DOTALL)
    
    # Add the correctly indented method to the class
    class_method = '''    def _create_xgboost(self):
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=50,  # Reduced from 100
            learning_rate=0.1,
            max_depth=3,     # Reduced from 5
            random_state=42,
            n_jobs=2,        # Limit parallel jobs
            subsample=0.8,   # Use 80% of data for training
            colsample_bytree=0.8  # Use 80% of features
        )'''
    
    # Find where to insert the method
    create_ridge_pos = modified_content.find('    def _create_ridge(self):')
    if create_ridge_pos > -1:
        # Insert before _create_ridge
        modified_content = modified_content[:create_ridge_pos] + class_method + '\n\n' + modified_content[create_ridge_pos:]
    else:
        # If _create_ridge isn't found, append to the end of the class before the train_model function
        train_model_pos = modified_content.find('def train_model(config=None):')
        if train_model_pos > -1:
            modified_content = modified_content[:train_model_pos] + class_method + '\n\n\n' + modified_content[train_model_pos:]
        else:
            print("Could not find appropriate position to insert _create_xgboost method")
            exit(1)
    
    # Write the fixed content back to the file
    with open('models/model_trainer.py', 'w') as f:
        f.write(modified_content)
    
    print("Fixed! The _create_xgboost method is now properly added to the ModelTrainer class.")
else:
    print("The _create_xgboost method appears to be correctly indented in the ModelTrainer class.") 