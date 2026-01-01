# Credit Card Fraud Detection

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)

## Introduction

This repository contains the source code for the Practical Project of the **Aprendizagem Aplicada à Segurança (AAS)** course.

The objective of this project is to detect credit card fraud by identifying suspicious transactions using machine learning models.

**Authors:**
* Henrique Coelho (108342)
* Paulo Macedo (102620)

## Prerequisities

* Python 3.8 or higher
* Pip (Python Package Installer)

## Installation

It is strongly recommended to run this project inside a virtual environment to avoid dependency conflicts.

### 1. Extract the compressed file
```bash
tar xvf credit-card-fraud-detection.tar.xz 
cd credit-card-fraud-detection
```

### 2. Create and Activate Virtual Environment

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset

This project utilizes the Credit Card Fraud Detection Dataset 2023.

  * **Source:** [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023?resource=download)
  * **Setup:** Place the dataset file in the root directory.

## Usage

We provide a Jupyter Notebook (`credit_card_fraud_detection.ipynb`) that contains the complete implementation of the credit card fraud detection project, so that its more organized and easier to follow.

Make sure to run/use the notebook in the environment where all dependencies are installed (select the right kernel).

## Major Results

Several machine learning models were evaluated for the task of credit card fraud detection, including Logistic Regression, Isolation Forest, and Random Forest. Model performance was assessed using an **80/20 train–test split**, as well as **k-fold cross-validation** to ensure robustness and generalization.

Among all tested models, **Random Forest** achieved the best overall performance and was selected as the final model due to its superior balance between precision and recall.

### Cross-Validation Results

A **5-fold cross-validation** was performed on the training set to evaluate model stability. The Random Forest classifier showed **consistently high performance across all folds**, with minimal variance, indicating strong generalization and low sensitivity to data partitioning.

- **Mean Cross-Validation F1-Score:** 99.88%
- **Standard Deviation:** < 0.01

These results confirm that the model’s performance is not dependent on a specific train–test split and is robust across different subsets of the data.

### Final Model Performance (Random Forest)

The following table summarizes the performance of the Random Forest model on the **held-out test set (20%)**:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Legitimate** | 0.998 | 0.999 | 0.999 | 56,863 |
| **Fraudulent** | 0.999 | 0.998 | 0.999 | 56,863 |
| **Accuracy** | | | **99.89%** | 113,726 |

### Overall Metrics

- **Accuracy:** 99.89%
- **Precision:** 99.94%
- **Recall:** 99.84%
- **F1-Score:** 99.89%
- **ROC-AUC:** 100.00%

### Key Findings

- The Random Forest model achieved **near-perfect classification performance**, effectively distinguishing between legitimate and fraudulent transactions.
- The **high recall** ensures a **very low False Negative Rate**, which is crucial in fraud detection scenarios where undetected fraud can have severe financial consequences.
- The **cross-validation results confirm the robustness and stability** of the model across different data splits.
- Supervised learning approaches significantly outperformed **unsupervised methods such as Isolation Forest** when labeled data was available.
- The final model is well-suited for real-world fraud detection systems where both accuracy and reliability are critical.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
