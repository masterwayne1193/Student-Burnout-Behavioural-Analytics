import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# --------------------------
# 1. Generate Synthetic Data
# --------------------------

n = 2000

data = pd.DataFrame({
    "lms_login_freq": np.random.normal(20, 5, n),  # logins per week
    "assignment_delay_days": np.random.normal(2, 1.5, n),
    "attendance_percent": np.random.normal(80, 10, n),
    "sentiment_score": np.random.uniform(-1, 1, n),
    "activity_irregularity": np.random.uniform(0, 1, n),
    "late_night_usage": np.random.uniform(0, 1, n)
})

# Behavioural Risk Logic
risk_score = (
    (30 - data["lms_login_freq"]) * 0.5 +
    data["assignment_delay_days"] * 5 +
    (100 - data["attendance_percent"]) * 0.3 +
    (-data["sentiment_score"] * 10) +
    data["activity_irregularity"] * 20 +
    data["late_night_usage"] * 15
)

data["risk_score"] = risk_score

# Burnout Classification
data["burnout_level"] = pd.cut(
    data["risk_score"],
    bins=[-100, 25, 50, 200],
    labels=["Low", "Medium", "High"]
)

# Dropout Probability (Binary)
data["dropout"] = (data["risk_score"] > 50).astype(int)

# --------------------------
# 2. Train Models
# --------------------------

X = data.drop(["burnout_level", "dropout"], axis=1)
y_burnout = data["burnout_level"]
y_dropout = data["dropout"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_burnout, test_size=0.2, random_state=42
)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

pred = rf.predict(X_test)
print("Burnout Classification Accuracy:", accuracy_score(y_test, pred))

# Logistic Regression for Dropout
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_scaled, y_dropout, test_size=0.2, random_state=42
)

log_reg = LogisticRegression()
log_reg.fit(X_train2, y_train2)

pred2 = log_reg.predict(X_test2)
print("Dropout Prediction Accuracy:", accuracy_score(y_test2, pred2))

# Decision Tree for Explainability
dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X_train, y_train)

print("Feature Importance (Random Forest):")
for feature, importance in zip(X.columns, rf.feature_importances_):
    print(feature, round(importance, 3))
