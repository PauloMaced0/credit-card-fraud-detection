"""
Data preprocessing module
Handles data cleaning, scaling, and splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE


class DataPreprocessor:
    """Class for preprocessing data before model training"""
    
    def __init__(self, df):
        """
        Initialize DataPreprocessor with dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        self.df = df.copy()
        self.scaler = None
        
    def remove_id_column(self):
        """Remove the 'id' column if it exists"""
        if 'id' in self.df.columns:
            print("Removing 'id' column...")
            self.df = self.df.drop('id', axis=1)
            print(f"✓ ID column removed. New shape: {self.df.shape}")
        return self.df
    
    def scale_amount(self):
        """Scale the Amount feature using RobustScaler"""
        print("\nScaling 'Amount' feature using RobustScaler...")
        print("(RobustScaler is robust to outliers)")
        
        self.scaler = RobustScaler()
        self.df['Amount'] = self.scaler.fit_transform(self.df[['Amount']])
        
        print("✓ Amount feature scaled successfully")
        return self.df, self.scaler
    
    def split_data(self):
        """
        Split data into train and test sets
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("\n" + "="*60)
        print("SPLITTING DATA")
        print("="*60)
        
        # Separate features and target
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, 
            stratify=y
        )
        
        print(f"\nTrain set size: {len(X_train):,} ({(1-TEST_SIZE)*100:.0f}%)")
        print(f"Test set size:  {len(X_test):,} ({TEST_SIZE*100:.0f}%)")
        
        # Verify class distribution is maintained
        print("\nClass distribution maintained:")
        print(f"  Train - Legitimate: {(y_train==0).sum():,}, Fraudulent: {(y_train==1).sum():,}")
        print(f"  Test  - Legitimate: {(y_test==0).sum():,}, Fraudulent: {(y_test==1).sum():,}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess(self):
        """
        Execute full preprocessing pipeline
        
        Returns:
            tuple: X_train, X_test, y_train, y_test, scaler
        """
        print("="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        # Step 1: Remove ID column
        self.remove_id_column()
        
        # Step 2: Scale Amount feature
        self.df, self.scaler = self.scale_amount()
        
        # Step 3: Split data
        X_train, X_test, y_train, y_test = self.split_data()
        
        print("\n✓ Preprocessing complete!")
        
        return X_train, X_test, y_train, y_test, self.scaler