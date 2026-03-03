# ==========================================
# AI-Powered Fraud Detection System (Fixed)
# Dataset: creditcard.csv
# ==========================================

# !pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ==========================================
# 1. Load Dataset
# ==========================================
data = pd.read_csv("/kaggle/input/datasets/organizations/mlg-ulb/creditcardfraud/creditcard.csv")

print("Dataset Shape:", data.shape)
print("\nClass Distribution:\n", data['Class'].value_counts())

# ==========================================
# 2. Feature & Target Split
# ==========================================
X = data.drop("Class", axis=1)
y = data["Class"]

# ==========================================
# 3. Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 4. Feature Scaling (Keep DataFrame format)
# ==========================================
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

# ==========================================
# 5. Handle Imbalance (SMOTE)
# ==========================================
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled, y_train
)

print("\nAfter SMOTE:\n", pd.Series(y_train_resampled).value_counts())

# ==========================================
# 6. Train XGBoost (Fixed)
# ==========================================
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='logloss'
)

model.fit(X_train_resampled, y_train_resampled)

# ==========================================
# 7. Predictions
# ==========================================
y_pred = model.predict(X_test_scaled)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

# ==========================================
# 8. Confusion Matrix
# ==========================================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================
# 9. Fraud Prediction Function (No Warning)
# ==========================================
def predict_fraud(transaction):
    transaction_df = pd.DataFrame([transaction], columns=X.columns)
    transaction_scaled = scaler.transform(transaction_df)
    prediction = model.predict(transaction_scaled)
    return "⚠️ Fraudulent Transaction" if prediction[0] == 1 else "✅ Legitimate Transaction"

# Example Prediction
sample_transaction = X.iloc[0].to_dict()
print("\nSample Prediction:", predict_fraud(sample_transaction))
