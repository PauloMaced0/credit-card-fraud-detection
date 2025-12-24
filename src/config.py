"""
Configuration file for Credit Card Fraud Detection Project
Contains all constants and configuration parameters
"""

# Random seed for reproducibility
RANDOM_STATE = 42

# File paths
DATA_PATH = '../data/creditcard_2023.csv'
MODEL_SAVE_DIR = '../models_saved/'
OUTPUT_DIR = '../outputs/'

# Data split ratio
TEST_SIZE = 0.2

# Model parameters
MODEL_PARAMS = {
    'logistic_regression': {
        'max_iter': 1000,
        'solver': 'lbfgs',
        'random_state': RANDOM_STATE
    },
    'decision_tree': {
        'max_depth': 10,
        'min_samples_split': 10,
        'random_state': RANDOM_STATE
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE
    },
    'isolation_forest': {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
        # contamination will be set dynamically based on fraud ratio
    }
}

# Cross-validation settings
CV_FOLDS = 5

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
COLOR_LEGITIMATE = '#2ecc71'
COLOR_FRAUDULENT = '#e74c3c'
DPI = 150

# Display settings
MAX_COLUMNS_DISPLAY = None