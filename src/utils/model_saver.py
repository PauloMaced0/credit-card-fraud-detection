"""
Model saving and loading module
Handles model persistence
"""

import joblib
import os
from config import MODEL_SAVE_DIR


class ModelSaver:
    """Class for saving and loading trained models"""
    
    def __init__(self, save_dir=MODEL_SAVE_DIR):
        """
        Initialize ModelSaver
        
        Args:
            save_dir (str): Directory to save models
        """
        self.save_dir = save_dir
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def save_model(self, model, model_name):
        """
        Save a trained model
        
        Args:
            model: Trained model object
            model_name (str): Name of the model
            
        Returns:
            str: Path where model was saved
        """
        filename = f'fraud_detection_model_{model_name.lower().replace(" ", "_")}.joblib'
        filepath = os.path.join(self.save_dir, filename)
        
        joblib.dump(model, filepath)
        print(f"✓ Model saved as '{filename}'")
        
        return filepath
        
    def save_scaler(self, scaler, scaler_name='amount_scaler'):
        """
        Save a scaler
        
        Args:
            scaler: Fitted scaler object
            scaler_name (str): Name for the scaler file
            
        Returns:
            str: Path where scaler was saved
        """
        filename = f'{scaler_name}.joblib'
        filepath = os.path.join(self.save_dir, filename)
        
        joblib.dump(scaler, filepath)
        print(f"✓ Scaler saved as '{filename}'")
        
        return filepath
        
    def load_model(self, model_filename):
        """
        Load a saved model
        
        Args:
            model_filename (str): Name of the model file
            
        Returns:
            Loaded model object
        """
        filepath = os.path.join(self.save_dir, model_filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        print(f"✓ Model loaded from '{model_filename}'")
        
        return model
        
    def load_scaler(self, scaler_filename='amount_scaler.joblib'):
        """
        Load a saved scaler
        
        Args:
            scaler_filename (str): Name of the scaler file
            
        Returns:
            Loaded scaler object
        """
        filepath = os.path.join(self.save_dir, scaler_filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
        
        scaler = joblib.load(filepath)
        print(f"✓ Scaler loaded from '{scaler_filename}'")
        
        return scaler
        
    def display_usage_instructions(self, model_filename):
        """
        Display instructions for using saved model
        
        Args:
            model_filename (str): Name of the saved model file
        """
        print("\n" + "="*60)
        print("HOW TO USE THE SAVED MODEL")
        print("="*60)
        print(f"""
# Load model and scaler
model = joblib.load('{model_filename}')
scaler = joblib.load('amount_scaler.joblib')

# Preprocess new data
new_data['Amount'] = scaler.transform(new_data[['Amount']])

# Make predictions
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]

# Interpret results
# 0 = Legitimate transaction
# 1 = Fraudulent transaction
        """)