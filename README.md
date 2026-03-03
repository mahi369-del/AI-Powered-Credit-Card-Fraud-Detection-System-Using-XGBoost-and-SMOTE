# AI-Powered-Credit-Card-Fraud-Detection-System-Using-XGBoost-and-SMOTE
AI-Powered Credit Card Fraud Detection System

## Project Overview

This project implements an end-to-end Machine Learning pipeline to detect fraudulent credit card transactions using the Kaggle Credit Card Fraud dataset. The system addresses the class imbalance problem using SMOTE and applies the XGBoost classifier to achieve high fraud detection performance.

## Objectives

- Detect fraudulent financial transactions
- Handle highly imbalanced datasets
- Maximize fraud recall while maintaining good precision
- Build a production-ready ML pipeline
- Generate business insights using evaluation metrics

## Dataset Information

Dataset: Credit Card Fraud Detection (Kaggle)

- Total Transactions: 284,807
- Fraud Cases: 492
- Features: 30 anonymized PCA features + Time + Amount
- Target Variable: `Class`
  - 0 → Legitimate
  - 1 → Fraud

This dataset is highly imbalanced (0.17% fraud cases).

## Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- Matplotlib & Seaborn

## System Architecture

1. Data Loading
2. Data Preprocessing
3. Train-Test Split (Stratified)
4. Feature Scaling (StandardScaler)
5. SMOTE Oversampling
6. XGBoost Model Training
7. Model Evaluation
8. Fraud Prediction Function

## Model Performance

- ROC-AUC Score: ~0.92
- Fraud Recall: ~85%
- Fraud Precision: ~58%
- Accuracy: ~99%

The model successfully detects most fraudulent transactions while maintaining acceptable false positives.

## Key Features

✔ Handles class imbalance using SMOTE  
✔ Uses XGBoost (industry-grade model)  
✔ Provides Confusion Matrix visualization  
✔ Calculates ROC-AUC score  
✔ Includes real-time fraud prediction function  

## Business Impact

- Reduces financial losses by detecting fraudulent transactions
- Minimizes false negatives (missed fraud)
- Supports risk management systems in FinTech
- Can be integrated into real-time transaction pipelines

## 🚀 How to Run

1. Download dataset from Kaggle
2. Place `creditcard.csv` in project directory
3. Install dependencies:
