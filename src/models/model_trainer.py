"""
Model training module
Handles training and evaluation of multiple ML models
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from config import MODEL_PARAMS


class ModelTrainer:
    """Class for training and evaluating machine learning models"""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize ModelTrainer with train and test data
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.trained_models = {}
        self.training_results = {}
        
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n" + "="*60)
        print("TRAINING: LOGISTIC REGRESSION")
        print("="*60)
        
        model = LogisticRegression(**MODEL_PARAMS['logistic_regression'])
        model.fit(self.X_train, self.y_train)
        
        results = self._evaluate_model(model, "Logistic Regression")
        
        self.trained_models['Logistic Regression'] = model
        self.training_results['Logistic Regression'] = results
        
        return model, results
    
    def train_decision_tree(self):
        """Train Decision Tree model"""
        print("\n" + "="*60)
        print("TRAINING: DECISION TREE")
        print("="*60)
        
        model = DecisionTreeClassifier(**MODEL_PARAMS['decision_tree'])
        model.fit(self.X_train, self.y_train)
        
        results = self._evaluate_model(model, "Decision Tree")
        
        self.trained_models['Decision Tree'] = model
        self.training_results['Decision Tree'] = results
        
        return model, results
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("TRAINING: RANDOM FOREST")
        print("="*60)
        
        model = RandomForestClassifier(**MODEL_PARAMS['random_forest'])
        model.fit(self.X_train, self.y_train)
        
        results = self._evaluate_model(model, "Random Forest")
        
        self.trained_models['Random Forest'] = model
        self.training_results['Random Forest'] = results
        
        return model, results
    
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("TRAINING: XGBOOST")
        print("="*60)
        
        model = XGBClassifier(**MODEL_PARAMS['xgboost'])
        model.fit(self.X_train, self.y_train)
        
        results = self._evaluate_model(model, "XGBoost")
        
        self.trained_models['XGBoost'] = model
        self.training_results['XGBoost'] = results
        
        return model, results
    
    def train_isolation_forest(self, y_full):
        """
        Train Isolation Forest (unsupervised anomaly detection)
        
        Args:
            y_full: Full target variable to calculate fraud ratio
        """
        print("\n" + "="*60)
        print("TRAINING: ISOLATION FOREST (Unsupervised)")
        print("="*60)
        
        # Calculate fraud ratio to set contamination parameter
        fraud_ratio = (y_full == 1).sum() / len(y_full)
        print(f"\nSetting contamination parameter to: {fraud_ratio:.4f}")
        print("(This tells the model the expected proportion of anomalies)")
        
        # Create model with dynamic contamination
        params = MODEL_PARAMS['isolation_forest'].copy()
        params['contamination'] = fraud_ratio
        model = IsolationForest(**params)
        model.fit(self.X_train)
        
        # Get predictions
        y_pred = model.predict(self.X_test)
        # Convert predictions: -1 (anomaly) -> 1 (fraud), 1 (normal) -> 0 (legitimate)
        y_pred = np.where(y_pred == -1, 1, 0)
        
        results = {
            'y_pred': y_pred,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': None  # Isolation Forest doesn't provide probabilities
        }
        
        print("\nIsolation Forest Results:")
        print("-"*40)
        print(f"Accuracy:  {results['accuracy']*100:.2f}%")
        print(f"Precision: {results['precision']*100:.2f}%")
        print(f"Recall:    {results['recall']*100:.2f}%")
        print(f"F1 Score:  {results['f1']*100:.2f}%")
        
        print("\nNote: Isolation Forest is unsupervised, so it doesn't use labels during training.")
        
        self.trained_models['Isolation Forest'] = model
        self.training_results['Isolation Forest'] = results
        
        return model, results
    
    def _evaluate_model(self, model, model_name):
        """
        Evaluate a trained model
        
        Args:
            model: Trained model
            model_name (str): Name of the model
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        results = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None
        }
        
        # Display results
        print(f"\n{model_name} Results:")
        print("-"*40)
        print(f"Accuracy:  {results['accuracy']*100:.2f}%")
        print(f"Precision: {results['precision']*100:.2f}%")
        print(f"Recall:    {results['recall']*100:.2f}%")
        print(f"F1 Score:  {results['f1']*100:.2f}%")
        if results['roc_auc']:
            print(f"ROC-AUC:   {results['roc_auc']*100:.2f}%")
        
        return results
    
    def train_all_models(self):
        """Train all models sequentially"""
        print("="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        self.train_logistic_regression()
        self.train_decision_tree()
        self.train_random_forest()
        self.train_xgboost()
        self.train_isolation_forest()
        
        print("\n✓ All models trained successfully!")
        
        return self.trained_models, self.training_results
    
    def perform_cross_validation(self, X, y):
        """
        Perform stratified k-fold cross-validation on all models
        
        Args:
            X: Full feature matrix
            y: Full target variable
            
        Returns:
            dict: Cross-validation results for each model
        """
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from config import CV_FOLDS
        
        print("\n" + "="*60)
        print(f"CROSS-VALIDATION ({CV_FOLDS}-Fold Stratified)")
        print("="*60)
        
        # Define cross-validation strategy
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=MODEL_PARAMS['random_forest']['random_state'])
        
        cv_results = {}
        
        print("\nPerforming cross-validation for each model...")
        print("(This may take a moment)\n")
        
        # Get untrained models
        models = {
            'Logistic Regression': LogisticRegression(**MODEL_PARAMS['logistic_regression']),
            'Decision Tree': DecisionTreeClassifier(**MODEL_PARAMS['decision_tree']),
            'Random Forest': RandomForestClassifier(**MODEL_PARAMS['random_forest']),
            'XGBoost': XGBClassifier(**MODEL_PARAMS['xgboost'])
        }
        
        for name, model in models.items():
            print(f"Cross-validating {name}...")
            
            # Calculate CV scores for multiple metrics
            accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
            recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')
            f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            roc_auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            
            cv_results[name] = {
                'Accuracy': (accuracy_scores.mean(), accuracy_scores.std()),
                'Precision': (precision_scores.mean(), precision_scores.std()),
                'Recall': (recall_scores.mean(), recall_scores.std()),
                'F1 Score': (f1_scores.mean(), f1_scores.std()),
                'ROC-AUC': (roc_auc_scores.mean(), roc_auc_scores.std())
            }
        
        print("\n✓ Cross-validation complete!")
        
        # Display results
        print("\nCross-Validation Results (Mean ± Std):")
        print("="*80)
        
        for name, results in cv_results.items():
            print(f"\n{name}:")
            print("-"*50)
            for metric, (mean, std) in results.items():
                print(f"  {metric:12s}: {mean*100:.2f}% (± {std*100:.2f}%)")
        
        print("\n" + "="*80)
        print("\nInterpretation:")
        print("-"*50)
        print("✓ Low standard deviation indicates consistent performance across folds")
        print("✓ High standard deviation suggests the model is sensitive to data splits")
        
        return cv_results
    
    def get_feature_importance(self):
        """
        Extract and display feature importance from tree-based models
        
        Returns:
            tuple: (rf_importance, xgb_importance) DataFrames
        """
        import pandas as pd
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Get feature names
        feature_names = self.X_train.columns
        
        # Random Forest feature importance
        rf_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.trained_models['Random Forest'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # XGBoost feature importance
        xgb_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.trained_models['XGBoost'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Features - Random Forest:")
        print("-"*40)
        print(rf_importance.head(10).to_string(index=False))
        
        print("\n\nTop 10 Features - XGBoost:")
        print("-"*40)
        print(xgb_importance.head(10).to_string(index=False))
        
        return rf_importance, xgb_importance
    
    def get_best_model(self):
        """
        Get the best performing model based on F1 score
        
        Returns:
            tuple: (best_model_name, best_model, best_results)
        """
        # Find best model based on F1 Score
        best_model_name = max(
            [k for k in self.training_results.keys() if k != 'Isolation Forest'],
            key=lambda x: self.training_results[x]['f1']
        )
        best_model = self.trained_models[best_model_name]
        best_results = self.training_results[best_model_name]
        
        print("="*60)
        print("BEST MODEL SELECTION")
        print("="*60)
        print(f"\nBest Model: {best_model_name}")
        print(f"   Selection Criterion: Highest F1 Score")
        print(f"   (F1 Score balances precision and recall - ideal for fraud detection)")
        
        print("\n\nFinal Performance Metrics:")
        print("-"*40)
        print(f"  Accuracy:  {best_results['accuracy']*100:.2f}%")
        print(f"  Precision: {best_results['precision']*100:.2f}%")
        print(f"  Recall:    {best_results['recall']*100:.2f}%")
        print(f"  F1 Score:  {best_results['f1']*100:.2f}%")
        if best_results['roc_auc']:
            print(f"  ROC-AUC:   {best_results['roc_auc']*100:.2f}%")
        
        return best_model_name, best_model, best_results