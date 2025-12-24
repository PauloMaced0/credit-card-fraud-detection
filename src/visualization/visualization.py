"""
Visualization module
Contains all plotting and visualization functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from config import COLOR_LEGITIMATE, COLOR_FRAUDULENT, DPI, OUTPUT_DIR


class Visualizer:
    """Class for creating all visualizations"""
    
    def __init__(self, df=None, output_dir=OUTPUT_DIR):
        """
        Initialize Visualizer
        
        Args:
            df (pd.DataFrame): Input dataframe
            output_dir (str): Directory to save plots
        """
        self.df = df
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_class_distribution(self, class_counts):
        """
        Plot class distribution as bar chart and pie chart
        
        Args:
            class_counts: Class value counts
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        colors = [COLOR_LEGITIMATE, COLOR_FRAUDULENT]
        bars = axes[0].bar(['Legitimate (0)', 'Fraudulent (1)'], class_counts.values, 
                          color=colors, edgecolor='black')
        axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Transactions', fontsize=12)
        axes[0].set_xlabel('Transaction Class', fontsize=12)
        
        # Add count labels on bars
        for bar, count in zip(bars, class_counts.values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                        f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Pie chart
        axes[1].pie(class_counts.values, labels=['Legitimate', 'Fraudulent'], 
                   autopct='%1.1f%%', colors=colors, explode=(0, 0.05),
                   shadow=True, startangle=90)
        axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}class_distribution.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        
    def plot_amount_distribution(self):
        """Plot transaction amount distribution by class"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Distribution by class
        for cls, color, label in [(0, COLOR_LEGITIMATE, 'Legitimate'), 
                                   (1, COLOR_FRAUDULENT, 'Fraudulent')]:
            subset = self.df[self.df['Class'] == cls]['Amount']
            axes[0].hist(subset, bins=50, alpha=0.6, color=color, label=label, density=True)
        
        axes[0].set_title('Transaction Amount Distribution by Class', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Amount ($)', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].legend()
        axes[0].set_xlim(0, self.df['Amount'].quantile(0.99))
        
        # Box plot
        self.df.boxplot(column='Amount', by='Class', ax=axes[1])
        axes[1].set_title('Amount Box Plot by Class', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Class (0=Legitimate, 1=Fraudulent)', fontsize=12)
        axes[1].set_ylabel('Amount ($)', fontsize=12)
        plt.suptitle('')  # Remove automatic title
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}amount_distribution.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        
    def plot_v_features_distribution(self, feature_importance_df):
        """
        Plot distribution of top 6 V features in 2x3 grid
        
        Args:
            feature_importance_df: DataFrame with features ranked by discriminative power
        """
        top_features = feature_importance_df['Feature'].head(6).tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top_features):
            for cls, color, label in [(0, COLOR_LEGITIMATE, 'Legitimate'), 
                                       (1, COLOR_FRAUDULENT, 'Fraudulent')]:
                subset = self.df[self.df['Class'] == cls][feature]
                axes[idx].hist(subset, bins=50, alpha=0.6, color=color, label=label, density=True)
            
            axes[idx].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Density')
            axes[idx].legend()
        
        plt.suptitle('Top 6 Most Discriminative Features', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}top_features_distribution.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        
    def plot_correlation_heatmap(self, correlation_matrix):
        """
        Plot correlation heatmap
        
        Args:
            correlation_matrix: Correlation matrix to plot
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', linewidths=0.5, cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Heatmap (Top Features + Amount + Class)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}correlation_heatmap.png', dpi=DPI, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_with_target(self, correlations):
        """
        Plot feature correlations with target variable
        
        Args:
            correlations: Series of correlation values with target
        """
        plt.figure(figsize=(12, 6))
        
        colors = [COLOR_FRAUDULENT if x < 0 else COLOR_LEGITIMATE for x in correlations.values]
        plt.barh(range(len(correlations)), correlations.values, color=colors)
        plt.yticks(range(len(correlations)), correlations.index)
        plt.xlabel('Correlation Coefficient', fontsize=12)
        plt.title('Feature Correlations with Fraud (Class)', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}correlation_with_target.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrices_grid(self, training_results, y_test):
        """
        Plot confusion matrices for all models in a 2x2 grid
        
        Args:
            training_results (dict): Dictionary of training results
            y_test: True test labels
        """
        from sklearn.metrics import confusion_matrix
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (model_name, results) in enumerate(training_results.items()):
            cm = confusion_matrix(y_test, results['y_pred'])
            
            # Calculate percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Create annotations with both count and percentage
            annotations = np.array([[f'{cm[i,j]:,}\n({cm_percent[i,j]:.1f}%)' 
                                    for j in range(2)] for i in range(2)])
            
            sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', ax=axes[idx],
                       xticklabels=['Legitimate', 'Fraud'],
                       yticklabels=['Legitimate', 'Fraud'],
                       cbar_kws={'label': 'Count'})
            
            axes[idx].set_title(f'{model_name}\nAccuracy: {results["accuracy"]*100:.2f}%', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Actual', fontsize=11)
            axes[idx].set_xlabel('Predicted', fontsize=11)
        
        plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}confusion_matrices.png', dpi=DPI, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm, model_name, cm_type='standard'):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            model_name (str): Name of the model
            cm_type (str): Type of confusion matrix ('standard' or 'isolation_forest')
        """
        plt.figure(figsize=(8, 6))
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both count and percentage
        annotations = np.array([
            [f'TN\n{cm[0,0]:,}\n({cm_percent[0,0]:.1f}%)', 
             f'FP\n{cm[0,1]:,}\n({cm_percent[0,1]:.1f}%)'],
            [f'FN\n{cm[1,0]:,}\n({cm_percent[1,0]:.1f}%)', 
             f'TP\n{cm[1,1]:,}\n({cm_percent[1,1]:.1f}%)']
        ])
        
        # Choose colormap based on type
        cmap = 'Oranges' if cm_type == 'isolation_forest' else 'Blues'
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap=cmap,
                   xticklabels=['Predicted\nLegitimate', 'Predicted\nFraudulent'],
                   yticklabels=['Actual\nLegitimate', 'Actual\nFraudulent'],
                   cbar_kws={'label': 'Count'})
        
        title_suffix = ' (Unsupervised)' if cm_type == 'isolation_forest' else ''
        plt.title(f'Confusion Matrix - {model_name}{title_suffix}', 
                 fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filename = f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
        plt.savefig(f'{self.output_dir}{filename}', dpi=DPI, bbox_inches='tight')
        plt.show()
        
    def plot_model_comparison(self, comparison_df):
        """
        Plot model comparison bar chart
        
        Args:
            comparison_df (pd.DataFrame): Comparison dataframe
        """
        # Extract numeric values
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            # Convert percentage strings to floats
            values = [float(x.strip('%')) for x in comparison_df[metric]]
            models = comparison_df['Model']
            
            bars = axes[idx].bar(range(len(models)), values, color='steelblue', edgecolor='black')
            axes[idx].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(f'{metric} (%)', fontsize=10)
            axes[idx].set_xticks(range(len(models)))
            axes[idx].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
            axes[idx].set_ylim(0, 105)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                             f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}model_comparison.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        
    def plot_roc_and_pr_curves(self, training_results, y_test):
        """
        Plot ROC and Precision-Recall curves for all models side by side
        
        Args:
            training_results (dict): Dictionary of training results
            y_test: True test labels
        """
        from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        # ROC Curve
        for idx, (model_name, results) in enumerate(training_results.items()):
            fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
            auc = roc_auc_score(y_test, results['y_pred_proba'])
            axes[0].plot(fpr, tpr, color=colors[idx], lw=2, 
                        label=f'{model_name} (AUC = {auc:.4f})')
        
        axes[0].plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0].set_ylabel('True Positive Rate', fontsize=12)
        axes[0].set_title('ROC Curves', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        for idx, (model_name, results) in enumerate(training_results.items()):
            precision, recall, _ = precision_recall_curve(y_test, results['y_pred_proba'])
            ap = average_precision_score(y_test, results['y_pred_proba'])
            axes[1].plot(recall, precision, color=colors[idx], lw=2, 
                        label=f'{model_name} (AP = {ap:.4f})')
        
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Recall', fontsize=12)
        axes[1].set_ylabel('Precision', fontsize=12)
        axes[1].set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        axes[1].legend(loc='lower left', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}roc_pr_curves.png', dpi=DPI, bbox_inches='tight')
        plt.show()
        
    def plot_roc_curve(self, fpr, tpr, roc_auc, model_name):
        """
        Plot ROC curve
        
        Args:
            fpr: False positive rate
            tpr: True positive rate
            roc_auc: ROC AUC score
            model_name (str): Name of the model
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        filename = f'{model_name.lower().replace(" ", "_")}_roc_curve.png'
        plt.savefig(f'{self.output_dir}{filename}', dpi=DPI, bbox_inches='tight')
        plt.show()
        
    def plot_precision_recall_curve(self, precision, recall, model_name):
        """
        Plot precision-recall curve
        
        Args:
            precision: Precision values
            recall: Recall values
            model_name (str): Name of the model
        """
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        filename = f'{model_name.lower().replace(" ", "_")}_pr_curve.png'
        plt.savefig(f'{self.output_dir}{filename}', dpi=DPI, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, rf_importance, xgb_importance):
        """
        Plot feature importance for Random Forest and XGBoost side by side
        
        Args:
            rf_importance: DataFrame with Random Forest feature importance
            xgb_importance: DataFrame with XGBoost feature importance
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        # Random Forest
        top_rf = rf_importance.head(15)
        axes[0].barh(range(len(top_rf)), top_rf['Importance'].values, color='#3498db')
        axes[0].set_yticks(range(len(top_rf)))
        axes[0].set_yticklabels(top_rf['Feature'].values)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Importance', fontsize=12)
        axes[0].set_title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
        
        # XGBoost
        top_xgb = xgb_importance.head(15)
        axes[1].barh(range(len(top_xgb)), top_xgb['Importance'].values, color='#9b59b6')
        axes[1].set_yticks(range(len(top_xgb)))
        axes[1].set_yticklabels(top_xgb['Feature'].values)
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Importance', fontsize=12)
        axes[1].set_title('XGBoost - Feature Importance', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}feature_importance.png', dpi=DPI, bbox_inches='tight')
        plt.show()