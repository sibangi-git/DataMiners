import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix,
    precision_recall_curve, average_precision_score, roc_curve
)
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv(r"C:\Users\sthfa\Downloads\cleaned_data 1.csv")

# Define features and target
features = ['PHYS14D','TOTINDA','SEX','AGE_G','BMI5CAT',
            'EDUCAG','INCOMG1','RFSMOK3','DRNKANY6','SSBSUGR2_CAT']
target = 'MICHD'

# Clean and encode
df_filtered = df[features + [target]].dropna().copy()
df_filtered[target] = df_filtered[target].map({1: 1, 2: 0}).astype(int)
order_map = {'Low': 1, 'Medium': 2, 'High': 3}
df_filtered['SSBSUGR2_CAT'] = (
    df_filtered['SSBSUGR2_CAT']
    .astype(str).str.strip().str.title()
    .map(order_map)
)
df_filtered = df_filtered.dropna(subset=['SSBSUGR2_CAT'])
df_filtered['SSBSUGR2_CAT'] = df_filtered['SSBSUGR2_CAT'].astype(int)

# Split features and target
X = df_filtered.drop(columns=[target])
y = df_filtered[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Base models
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=0.7,
    scale_pos_weight=10,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

lr_model = LogisticRegression(max_iter=1000, solver='liblinear')

# Stacked ensemble
stacked_model = StackingClassifier(
    estimators=[('xgb', xgb_model)],
    final_estimator=lr_model,
    passthrough=True,
    cv=3
)
stacked_model.fit(X_train_res, y_train_res)

# Predict probabilities
y_prob = stacked_model.predict_proba(X_test_scaled)[:, 1]

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"Stacked Model (AUC = {roc_auc:.3f})", color='purple')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Stacked Ensemble")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Precision-Recall curve
# precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
# avg_precision = average_precision_score(y_test, y_prob)

# plt.figure(figsize=(8,6))
# plt.plot(recall, precision, label=f"PR Curve (AP = {avg_precision:.3f})", color='blue')
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()

# Threshold tuning
# for threshold in [0.45, 0.5, 0.55, 0.6]:
#     y_pred = (y_prob >= threshold).astype(int)
#     acc = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)
#     print(f"\n🔍 Threshold = {threshold}")
#     print(f"Accuracy : {acc:.3f}")
#     print(f"F1-score : {f1:.3f}")
#     print("Confusion matrix [TN FP; FN TP]:")
#     print(cm)
threshold = 0.45
y_pred = (y_prob >= threshold).astype(int)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"\n🔍 Threshold = {threshold}")
print(f"Accuracy : {acc:.3f}")
print(f"F1-score : {f1:.3f}")
print("Confusion matrix [TN FP; FN TP]:")
print(cm)
# # Feature importance from XGBoost
# xgb_model.fit(X_train_res, y_train_res)
# importances = xgb_model.feature_importances_
# feature_names = X.columns
# feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# feat_df = feat_df.sort_values(by='Importance', ascending=False)

# plt.figure(figsize=(8,5))
# sns.barplot(data=feat_df.head(10), x='Importance', y='Feature', palette='viridis')
# plt.title("Top 10 Feature Importances (XGBoost)")
# plt.tight_layout()
# plt.show()
import joblib

# # Save scaler and model separately
joblib.dump(scaler, "scaler_logistic_R00.pkl")
joblib.dump(stacked_model, "logistic_model_R00.pkl")..add a logicstic model, decision tree and make comparision with this model based on print(f"Accuracy : {acc:.3f}")
print(f"F1-score : {f1:.3f}")
print("Confusion matrix [TN FP; FN TP]:")
..show it in a table form..ignore all the comment part.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix,
    precision_recall_curve, average_precision_score, roc_curve
)
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv(r"C:\Users\sthfa\Downloads\cleaned_data 1.csv")

# Define features and target
features = ['PHYS14D','TOTINDA','SEX','AGE_G','BMI5CAT',
            'EDUCAG','INCOMG1','RFSMOK3','DRNKANY6','SSBSUGR2_CAT']
target = 'MICHD'

# Clean and encode
df_filtered = df[features + [target]].dropna().copy()
df_filtered[target] = df_filtered[target].map({1: 1, 2: 0}).astype(int)
order_map = {'Low': 1, 'Medium': 2, 'High': 3}
df_filtered['SSBSUGR2_CAT'] = (
    df_filtered['SSBSUGR2_CAT']
    .astype(str).str.strip().str.title()
    .map(order_map)
)
df_filtered = df_filtered.dropna(subset=['SSBSUGR2_CAT'])
df_filtered['SSBSUGR2_CAT'] = df_filtered['SSBSUGR2_CAT'].astype(int)

# Split features and target
X = df_filtered.drop(columns=[target])
y = df_filtered[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Base models
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=0.7,
    scale_pos_weight=10,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

lr_model = LogisticRegression(max_iter=1000, solver='liblinear')

# Stacked ensemble
stacked_model = StackingClassifier(
    estimators=[('xgb', xgb_model)],
    final_estimator=lr_model,
    passthrough=True,
    cv=3
)
stacked_model.fit(X_train_res, y_train_res)

# Predict probabilities
y_prob = stacked_model.predict_proba(X_test_scaled)[:, 1]

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"Stacked Model (AUC = {roc_auc:.3f})", color='purple')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Stacked Ensemble")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Precision-Recall curve
# precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
# avg_precision = average_precision_score(y_test, y_prob)

# plt.figure(figsize=(8,6))
# plt.plot(recall, precision, label=f"PR Curve (AP = {avg_precision:.3f})", color='blue')
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()

# Threshold tuning
# for threshold in [0.45, 0.5, 0.55, 0.6]:
#     y_pred = (y_prob >= threshold).astype(int)
#     acc = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)
#     print(f"\n🔍 Threshold = {threshold}")
#     print(f"Accuracy : {acc:.3f}")
#     print(f"F1-score : {f1:.3f}")
#     print("Confusion matrix [TN FP; FN TP]:")
#     print(cm)
threshold = 0.45
y_pred = (y_prob >= threshold).astype(int)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"\n🔍 Threshold = {threshold}")
print(f"Accuracy : {acc:.3f}")
print(f"F1-score : {f1:.3f}")
print("Confusion matrix [TN FP; FN TP]:")
print(cm)
# # Feature importance from XGBoost
# xgb_model.fit(X_train_res, y_train_res)
# importances = xgb_model.feature_importances_
# feature_names = X.columns
# feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# feat_df = feat_df.sort_values(by='Importance', ascending=False)

# plt.figure(figsize=(8,5))
# sns.barplot(data=feat_df.head(10), x='Importance', y='Feature', palette='viridis')
# plt.title("Top 10 Feature Importances (XGBoost)")
# plt.tight_layout()
# plt.show()
# import joblib

# # # Save scaler and model separately
# joblib.dump(scaler, "scaler_logistic_R00.pkl")
# joblib.dump(stacked_model, "logistic_model_R00.pkl")
