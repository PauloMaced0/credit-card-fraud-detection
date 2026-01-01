"""
Data loading and initial inspection module
"""

import pandas as pd
import numpy as np
from config import MAX_COLUMNS_DISPLAY


class DataLoader:
    """Class to handle data loading and basic information display"""
    
    def __init__(self, filepath):
        """
        Initialize DataLoader with file path
        
        Args:
            filepath (str): Path to the CSV file
        """
        self.filepath = filepath
        self.df = None
        
    def load_data(self):
        """Load the dataset from CSV file"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.filepath)
        print(f"Dataset loaded successfully! Shape: {self.df.shape}")
        return self.df
    
    def display_overview(self):
        """Display basic dataset information"""
        print("="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        print(f"\nDataset Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"\nMemory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
    def display_sample(self, n=5):
        """
        Display first n rows of dataset
        
        Args:
            n (int): Number of rows to display
        """
        print(f"\nFirst {n} rows of the dataset:")
        return self.df.head(n)
    
    def display_info(self):
        """Display data types and non-null counts"""
        print("\nData Types and Non-Null Counts:")
        print("-"*40)
        self.df.info()
        
    def display_statistics(self):
        """Display statistical summary"""
        print("\nStatistical Summary:")
        return self.df.describe()
    
    def check_data_quality(self):
        """Perform comprehensive data quality checks"""
        print("="*60)
        print("DATA QUALITY ASSESSMENT")
        print("="*60)
        
        # Check for missing values
        self._check_missing_values()
        
        # Check for duplicates
        self._check_duplicates()
        
        # Check unique IDs
        self._check_unique_ids()
    
    def _check_missing_values(self):
        """Check for missing values in dataset"""
        print("\n1. MISSING VALUES ANALYSIS")
        print("-"*40)
        
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        
        if missing_values.sum() == 0:
            print("✓ No missing values found in the dataset!")
        else:
            print("Missing values found:")
            missing_df = pd.DataFrame({
                'Missing Count': missing_values[missing_values > 0],
                'Percentage': missing_percentage[missing_values > 0]
            })
            print(missing_df)
    
    def _check_duplicates(self):
        """Check for duplicate records"""
        print("\n2. DUPLICATE RECORDS")
        print("-"*40)
        
        duplicates = self.df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicates:,}")
        print(f"Percentage of duplicates: {(duplicates/len(self.df))*100:.2f}%")
    
    def _check_unique_ids(self):
        """Check if IDs are unique"""
        print("\n3. ID UNIQUENESS")
        print("-"*40)
        
        if 'id' in self.df.columns:
            unique_ids = self.df['id'].nunique()
            print(f"Total records: {len(self.df):,}")
            print(f"Unique IDs: {unique_ids:,}")
            print(f"IDs are unique: {unique_ids == len(self.df)}")
        else:
            print("No 'id' column found in dataset")
    
    def analyze_class_distribution(self):
        """Analyze and display class distribution"""
        print("="*60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Calculate class distribution
        class_counts = self.df['Class'].value_counts()
        class_percentages = self.df['Class'].value_counts(normalize=True) * 100
        
        print("\nClass Distribution:")
        print("-"*40)
        print(f"Legitimate Transactions (Class 0): {class_counts[0]:,} ({class_percentages[0]:.2f}%)")
        print(f"Fraudulent Transactions (Class 1): {class_counts[1]:,} ({class_percentages[1]:.2f}%)")
        print(f"\nClass Ratio (Legitimate:Fraudulent): {class_counts[0]/class_counts[1]:.2f}:1")
        
        # Determine if balanced
        imbalance_ratio = class_counts[0] / class_counts[1]
        if 0.8 <= class_percentages[1]/class_percentages[0] <= 1.2:
            balance_status = "BALANCED"
            balance_note = """The dataset is balanced, meaning both classes have similar representation.
This simplifies our modeling approach as we don't need to apply special techniques
like SMOTE (Synthetic Minority Over-sampling Technique) or class weights."""
        else:
            balance_status = "IMBALANCED"
            balance_note = "The dataset is imbalanced. Consider using techniques like SMOTE or class weights."
        
        print(f"\nDataset Status: {balance_status}")
        print(f"\nNote: {balance_note}")
        
        return class_counts, class_percentages