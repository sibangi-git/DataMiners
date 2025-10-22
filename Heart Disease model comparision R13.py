# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\sthfa\Downloads\cleaned_data 1.csv")
# Define features and target
features = ['PHYS14D','TOTINDA','SEX','AGE_G','BMI5CAT',
            'EDUCAG','INCOMG1','RFSMOK3','DRNKANY6','SSBSUGR2_CAT']
target = 'MICHD'

# Clean and encode data
df_filtered = df[features + [target]].dropna().copy()
df_filtered[target] = df_filtered[target].map({1: 1, 2: 0}).astype(int)

# MANUAL ORDINAL ENCODING for SSBSUGR2_CAT (Low < Medium < High)
order_map = {'Low': 1, 'Medium': 2, 'High': 3}
df_filtered['SSBSUGR2_CAT'] = (
    df_filtered['SSBSUGR2_CAT'].astype(str).str.strip().str.title().map(order_map)
)

# Defining X and y
X = df_filtered.drop(columns=[target])
y = df_filtered[target]

# Split features and target into training and testing sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Parameters set after hyperparameter tuning
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

# Other base models
lr_model = LogisticRegression(max_iter=1000, solver='liblinear')
dt_model = DecisionTreeClassifier(random_state=42, max_depth=None, min_samples_split=2)

# Create Stacked Ensemble Model
stacked_model = StackingClassifier(
    estimators=[('xgb', xgb_model)],
    final_estimator=lr_model,
    passthrough=True,
    cv=3
)
# Fit stacked model
stacked_model.fit(X_train_res, y_train_res)

# Fit individual models for comparison
lr_model.fit(X_train_res, y_train_res)
dt_model.fit(X_train_res, y_train_res)

# Set threshold for classification after threshold tuning
threshold = 0.45

# Function to evaluate models
def eval_model(name, model, X_te, y_te):
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "Model": name,
        "Accuracy": round(acc, 3),
        "F1": round(f1, 3),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }, y_prob

# Evaluate all models
stack_metrics, stack_prob = eval_model("Stacked (XGB → LR)", stacked_model, X_test_scaled, y_test)
lr_metrics, lr_prob = eval_model("Logistic Regression", lr_model, X_test_scaled, y_test)
dt_metrics, dt_prob = eval_model("Decision Tree", dt_model, X_test_scaled, y_test)

# Compile results
results_df = pd.DataFrame([stack_metrics, lr_metrics, dt_metrics]).set_index("Model")
print(results_df)

# ROC curve for stacked model
# fpr, tpr, _ = roc_curve(y_test, stack_prob)
# roc_auc = roc_auc_score(y_test, stack_prob)
# plt.figure(figsize=(8,6))
# plt.plot(fpr, tpr, label=f"Stacked Model (AUC = {roc_auc:.3f})")
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve - Stacked Ensemble")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()

# Save scaler and models
# joblib.dump(scaler, "scaler_R00.pkl")
# joblib.dump(stacked_model, "stacked_model_R00.pkl")
# joblib.dump(lr_model, "logreg_model_R00.pkl")
# joblib.dump(dt_model, "decision_tree_model_R00.pkl")
