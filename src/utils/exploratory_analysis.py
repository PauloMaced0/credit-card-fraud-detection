"""
Exploratory Data Analysis module
Contains functions for analyzing features and distributions
"""

import pandas as pd
import numpy as np


class ExploratoryAnalyzer:
    """Class for performing exploratory data analysis"""
    
    def __init__(self, df):
        """
        Initialize ExploratoryAnalyzer with dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        self.df = df
        
    def analyze_amount_feature(self):
        """Analyze the Amount feature by class"""
        print("="*60)
        print("AMOUNT FEATURE ANALYSIS")
        print("="*60)
        
        print("\nAmount Statistics by Class:")
        print("-"*40)
        amount_stats = self.df.groupby('Class')['Amount'].describe()
        print(amount_stats)
        
        # Compare mean and median
        print("\n\nKey Observations:")
        print("-"*40)
        legit_mean = self.df[self.df['Class']==0]['Amount'].mean()
        fraud_mean = self.df[self.df['Class']==1]['Amount'].mean()
        legit_median = self.df[self.df['Class']==0]['Amount'].median()
        fraud_median = self.df[self.df['Class']==1]['Amount'].median()
        
        print(f"Legitimate - Mean: ${legit_mean:.2f}, Median: ${legit_median:.2f}")
        print(f"Fraudulent - Mean: ${fraud_mean:.2f}, Median: ${fraud_median:.2f}")
        
        return amount_stats
    
    def analyze_v_features(self):
        """Analyze V features - Calculate discriminative power"""
        v_features = [f'V{i}' for i in range(1, 29)]
        
        # Calculate mean differences between classes
        mean_diff = []
        for feature in v_features:
            fraud_mean = self.df[self.df['Class']==1][feature].mean()
            legit_mean = self.df[self.df['Class']==0][feature].mean()
            mean_diff.append(abs(fraud_mean - legit_mean))
        
        # Sort features by discriminative power
        feature_importance_df = pd.DataFrame({
            'Feature': v_features,
            'Mean_Difference': mean_diff
        }).sort_values('Mean_Difference', ascending=False)
        
        print("="*60)
        print("TOP DISCRIMINATIVE FEATURES (by mean difference)")
        print("="*60)
        print("\nTop 10 Features:")
        print(feature_importance_df.head(10).to_string(index=False))
        
        return feature_importance_df
    
    def calculate_correlation_with_target(self):
        """Calculate correlation of each feature with the target variable"""
        # Drop id column if it exists
        df_temp = self.df.drop('id', axis=1) if 'id' in self.df.columns else self.df
        
        # Calculate correlations with target
        correlations = df_temp.corr()['Class'].drop('Class').sort_values(key=abs, ascending=False)
        
        print("="*60)
        print("CORRELATION WITH TARGET (Class)")
        print("="*60)
        print("\nTop 10 features most correlated with fraud:")
        print("-"*40)
        print(correlations.head(10))
        
        print("\n\nBottom 10 features least correlated with fraud:")
        print("-"*40)
        print(correlations.tail(10))
        
        return correlations
    
    def calculate_correlations(self):
        """Calculate correlation matrix for top features"""
        # Get top 10 V features by discriminative power
        feature_importance = self.analyze_v_features()
        top_features = feature_importance.head(10)['Feature'].tolist()
        
        # Add Amount and Class
        features_to_correlate = top_features + ['Amount', 'Class']
        
        # Calculate correlation matrix
        correlation_matrix = self.df[features_to_correlate].corr()
        
        return correlation_matrix, top_features