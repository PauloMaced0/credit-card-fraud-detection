"""
Model evaluation module
Provides detailed evaluation metrics and analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, precision_recall_curve
)


class ModelEvaluator:
    """Class for detailed model evaluation"""
    
    def __init__(self, y_test):
        """
        Initialize ModelEvaluator
        
        Args:
            y_test: True test labels
        """
        self.y_test = y_test
        
    def create_comparison_table(self, training_results):
        """
        Create a comparison table for all models
        
        Args:
            training_results (dict): Dictionary of training results
            
        Returns:
            pd.DataFrame: Comparison table
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_data = []
        for model_name, results in training_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']*100:.2f}%",
                'Precision': f"{results['precision']*100:.2f}%",
                'Recall': f"{results['recall']*100:.2f}%",
                'F1 Score': f"{results['f1']*100:.2f}%",
                'ROC-AUC': f"{results['roc_auc']*100:.2f}%" if results['roc_auc'] else "N/A"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        return comparison_df
    
    def display_classification_report(self, y_pred, model_name):
        """
        Display detailed classification report
        
        Args:
            y_pred: Model predictions
            model_name (str): Name of the model
        """
        print("\n" + "="*60)
        print(f"DETAILED CLASSIFICATION REPORT - {model_name}")
        print("="*60)
        
        print("\n" + classification_report(
            self.y_test, y_pred, 
            target_names=['Legitimate', 'Fraudulent']
        ))
        
        print("\nMetric Definitions:")
        print("-"*40)
        print("✓ Precision: Of all predicted frauds, what % are actually frauds?")
        print("✓ Recall: Of all actual frauds, what % did we catch?")
        print("✓ F1-Score: Harmonic mean of precision and recall")
        print("✓ Support: Number of actual occurrences of each class")
    
    def analyze_confusion_matrix(self, y_pred, model_name):
        """
        Analyze confusion matrix and display business impact
        
        Args:
            y_pred: Model predictions
            model_name (str): Name of the model
            
        Returns:
            tuple: (tn, fp, fn, tp)
        """
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\nBusiness Impact Analysis - {model_name}:")
        print("-"*40)
        print(f"Frauds Detected (TP): {tp:,} out of {tp+fn:,} ({tp/(tp+fn)*100:.2f}%)")
        print(f"Frauds Missed (FN): {fn:,} - These would result in financial losses")
        print(f"False Alarms (FP): {fp:,} - Legitimate transactions incorrectly blocked")
        print(f"Correct Approvals (TN): {tn:,} - Legitimate transactions correctly approved")

        print(f"\nDetection Rate: {(tp/(tp+fn))*100:.2f}% of frauds caught")
        print(f"False Alarm Rate: {(fp/(fp+tn))*100:.2f}% of legitimate flagged")
        
        return tn, fp, fn, tp
    
    def calculate_roc_curve(self, y_pred_proba):
        """
        Calculate ROC curve data
        
        Args:
            y_pred_proba: Predicted probabilities
            
        Returns:
            tuple: (fpr, tpr, thresholds)
        """
        if y_pred_proba is not None:
            fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
            return fpr, tpr, thresholds
        return None, None, None
    
    def calculate_precision_recall_curve(self, y_pred_proba):
        """
        Calculate precision-recall curve data
        
        Args:
            y_pred_proba: Predicted probabilities
            
        Returns:
            tuple: (precision, recall, thresholds)
        """
        if y_pred_proba is not None:
            precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred_proba)
            return precision, recall, thresholds
        return None, None, None