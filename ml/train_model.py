import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import os

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Features & target
X = data.drop(columns=["recommended_crop"])
y = data["recommended_crop"]

# Encode categorical column (previous_crop)
label_encoder_prev = LabelEncoder()
X["previous_crop"] = label_encoder_prev.fit_transform(X["previous_crop"])

# Encode target column
label_encoder_target = LabelEncoder()
y = label_encoder_target.fit_transform(y)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

print("Before SMOTE:", pd.Series(y).value_counts())
print("After SMOTE:", pd.Series(y_res).value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Train model (XGBoost)
model = XGBClassifier(eval_metric="mlogloss", random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(model, X_res, y_res, cv=5)
print("CV Accuracy:", scores.mean())

# Save model + scaler + encoders
joblib.dump(model, "models/crop_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder_prev, "models/prev_encoder.pkl")
joblib.dump(label_encoder_target, "models/label_encoder.pkl")

print("✅ Model, scaler, and encoders saved in 'models/' folder")
