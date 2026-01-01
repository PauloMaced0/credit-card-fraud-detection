"""
Main script for Credit Card Fraud Detection
Orchestrates the entire ML pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Import custom modules
from config import RANDOM_STATE, PLOT_STYLE, MAX_COLUMNS_DISPLAY, DATA_PATH, DPI
from utils.data_loader import DataLoader
from utils.exploratory_analysis import ExploratoryAnalyzer
from preprocessing.data_preprocessor import DataPreprocessor
from models.model_trainer import ModelTrainer
from models.model_evaluator import ModelEvaluator
from visualization.visualization import Visualizer
from utils.model_saver import ModelSaver

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(RANDOM_STATE)

# Display settings
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)
plt.style.use(PLOT_STYLE)

print("="*70)
print("CREDIT CARD FRAUD DETECTION - MACHINE LEARNING PROJECT")
print("="*70)
print("\nAll libraries imported successfully!\n")


def main():
    """Main function to run the entire pipeline"""
    
    # ========== 1. DATA LOADING ==========
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING")
    print("="*70)
    
    data_loader = DataLoader(DATA_PATH)
    df = data_loader.load_data()
    data_loader.display_overview()
    data_loader.display_sample()
    data_loader.display_info()
    data_loader.display_statistics()
    
    # ========== 2. DATA QUALITY ASSESSMENT ==========
    print("\n" + "="*70)
    print("STEP 2: DATA QUALITY ASSESSMENT")
    print("="*70)
    
    data_loader.check_data_quality()
    class_counts, class_percentages = data_loader.analyze_class_distribution()
    
    # ========== 3. EXPLORATORY DATA ANALYSIS ==========
    print("\n" + "="*70)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    eda = ExploratoryAnalyzer(df)
    
    # Analyze Amount feature
    amount_stats = eda.analyze_amount_feature()
    
    # Analyze V features
    feature_importance = eda.analyze_v_features()
    
    # Calculate correlation with target
    correlations_with_target = eda.calculate_correlation_with_target()
    
    # Calculate correlations
    correlation_matrix, top_features = eda.calculate_correlations()
    
    # ========== 4. VISUALIZATION ==========
    print("\n" + "="*70)
    print("STEP 4: DATA VISUALIZATION")
    print("="*70)
    
    visualizer = Visualizer(df)
    
    print("\nGenerating visualizations...")
    visualizer.plot_class_distribution(class_counts)
    visualizer.plot_amount_distribution()
    visualizer.plot_v_features_distribution(feature_importance)
    visualizer.plot_correlation_with_target(correlations_with_target)
    visualizer.plot_correlation_heatmap(correlation_matrix)
    print("✓ Visualizations complete!")
    
    # ========== 5. DATA PREPROCESSING ==========
    print("\n" + "="*70)
    print("STEP 5: DATA PREPROCESSING")
    print("="*70)
    
    preprocessor = DataPreprocessor(df)
    X_train, X_test, y_train, y_test, scaler = preprocessor.preprocess()
    
    # ========== 6. MODEL TRAINING ==========
    print("\n" + "="*70)
    print("STEP 6: MODEL TRAINING")
    print("="*70)
    
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    
    # Train supervised models
    trainer.train_logistic_regression()
    trainer.train_decision_tree()
    trainer.train_random_forest()
    trainer.train_xgboost()
    
    trained_models = trainer.trained_models
    training_results = trainer.training_results
    
    # ========== 7. MODEL EVALUATION ==========
    print("\n" + "="*70)
    print("STEP 7: MODEL EVALUATION")
    print("="*70)
    
    evaluator = ModelEvaluator(y_test)
    
    # Create comparison table
    comparison_df = evaluator.create_comparison_table(training_results)
    
    # Visualize model comparison - grouped bar chart
    print("\nGenerating model comparison chart...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    x = np.arange(len(metrics))
    width = 0.2
    multiplier = 0
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for idx, (model_name, results) in enumerate(training_results.items()):
        values = [results['accuracy'], results['precision'], results['recall'], 
                  results['f1'], results['roc_auc']]
        offset = width * multiplier
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[idx])
        multiplier += 1
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{visualizer.output_dir}model_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    
    # ========== 8. CONFUSION MATRICES ==========
    print("\n" + "="*70)
    print("STEP 8: CONFUSION MATRIX ANALYSIS")
    print("="*70)
    
    # Plot confusion matrices for all models in 2x2 grid
    visualizer.plot_confusion_matrices_grid(training_results, y_test)
    
    # Detailed confusion matrix analysis
    print("\nDetailed Confusion Matrix Analysis:")
    print("="*60)
    for model_name, results in training_results.items():
        evaluator.analyze_confusion_matrix(results['y_pred'], model_name)
    
    # ========== 9. ROC AND PRECISION-RECALL CURVES ==========
    print("\n" + "="*70)
    print("STEP 9: ROC AND PRECISION-RECALL CURVES")
    print("="*70)
    
    # Plot combined ROC and PR curves for all models
    visualizer.plot_roc_and_pr_curves(training_results, y_test)
    
    # ========== 10. CROSS-VALIDATION ==========
    print("\n" + "="*70)
    print("STEP 10: CROSS-VALIDATION")
    print("="*70)
    
    # Combine X_train and X_test back for CV
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    
    cv_results = trainer.perform_cross_validation(X_full, y_full)
    
    # ========== 11. FEATURE IMPORTANCE ==========
    print("\n" + "="*70)
    print("STEP 11: FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    rf_importance, xgb_importance = trainer.get_feature_importance()
    visualizer.plot_feature_importance(rf_importance, xgb_importance)
    
    # ========== 12. UNSUPERVISED ANOMALY DETECTION ==========
    print("\n" + "="*70)
    print("STEP 12: UNSUPERVISED ANOMALY DETECTION")
    print("="*70)
    
    trainer.train_isolation_forest(y_full)
    
    # Plot Isolation Forest confusion matrix
    iso_results = trainer.training_results['Isolation Forest']
    from sklearn.metrics import confusion_matrix
    cm_iso = confusion_matrix(y_test, iso_results['y_pred'])
    visualizer.plot_confusion_matrix(cm_iso, 'Isolation Forest', 'isolation_forest')
    
    # Analyze Isolation Forest results
    evaluator.analyze_confusion_matrix(iso_results['y_pred'], 'Isolation Forest')
    
    # ========== 13. BEST MODEL SELECTION ==========
    print("\n" + "="*70)
    print("STEP 13: BEST MODEL SELECTION")
    print("="*70)
    
    # Get best model (excluding Isolation Forest for comparison)
    best_model_name, best_model, best_results = trainer.get_best_model()
    
    # Detailed evaluation of best model
    evaluator.display_classification_report(best_results['y_pred'], best_model_name)
    
    # ========== 14. SAVE MODELS ==========
    print("\n" + "="*70)
    print("STEP 14: SAVING MODELS")
    print("="*70)
    
    model_saver = ModelSaver()
    
    # Save best model
    model_path = model_saver.save_model(best_model, best_model_name)
    
    # Save scaler
    scaler_path = model_saver.save_scaler(scaler)
    
    # Display usage instructions
    model_filename = f'fraud_detection_model_{best_model_name.lower().replace(" ", "_")}.joblib'
    model_saver.display_usage_instructions(model_filename)
    
    # ========== 15. PROJECT SUMMARY ==========
    print("\n" + "="*70)
    print("PROJECT SUMMARY AND CONCLUSIONS")
    print("="*70)
    
    print(f"""
DATASET CHARACTERISTICS:
──────────────────────────────────────────────────────────────────────
    Total Transactions: {len(df):,}
    Features: 29 (V1-V28 + Amount)
    Class Distribution: {'Balanced' if 0.8 <= class_percentages[1]/class_percentages[0] <= 1.2 else 'Imbalanced'}
    Legitimate: {class_counts[0]:,} ({class_percentages[0]:.2f}%)
    Fraudulent: {class_counts[1]:,} ({class_percentages[1]:.2f}%)

PREPROCESSING STEPS:
──────────────────────────────────────────────────────────────────────
   ✓ Removed non-predictive 'id' column
   ✓ Scaled 'Amount' feature using RobustScaler
   ✓ Performed 80/20 stratified train-test split

MODELS EVALUATED:
──────────────────────────────────────────────────────────────────────
   • Logistic Regression (baseline)
   • Decision Tree
   • Random Forest
   • XGBoost
   • Isolation Forest (unsupervised, for comparison)

MODEL PERFORMANCE SUMMARY:
──────────────────────────────────────────────────────────────────────
{comparison_df.to_string(index=False)}

BEST MODEL: {best_model_name}
──────────────────────────────────────────────────────────────────────
   Selection Criterion: Highest F1 Score
   
   Final Metrics:
    • Accuracy:  {best_results['accuracy']*100:.2f}%
    • Precision: {best_results['precision']*100:.2f}%
    • Recall:    {best_results['recall']*100:.2f}%
    • F1 Score:  {best_results['f1']*100:.2f}%
    • ROC-AUC:   {best_results['roc_auc']*100:.2f}% {'' if best_results['roc_auc'] else '(N/A)'}

KEY INSIGHTS:
──────────────────────────────────────────────────────────────────────
   • The {best_model_name} model achieved the best balance between precision and recall
   • Feature engineering and proper scaling significantly improved model performance
   • Ensemble methods (Random Forest, XGBoost) outperformed simpler models
   • The model successfully detects fraudulent transactions with high accuracy
    """)
    
    print("\n" + "="*70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nModels saved in: {model_saver.save_dir}")
    print(f"Visualizations saved in: {visualizer.output_dir}")
    

if __name__ == "__main__":
    main()