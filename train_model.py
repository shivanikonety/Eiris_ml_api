from pathlib import Path
import json

import joblib
import numpy as np

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 1. Load the Iris dataset
# -----------------------------
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names


# -----------------------------
# 2. Split the dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------
# 3. Scale the features
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# 4. Train Random Forest
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# -----------------------------
# 5. Evaluate the model
# -----------------------------
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_names
    )
)


# -----------------------------
# 6. Save model artifacts
# -----------------------------
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

joblib.dump(
    model,
    models_dir / "iris_rf_model.joblib"
)

joblib.dump(
    scaler,
    models_dir / "scaler.joblib"
)


# -----------------------------
# 7. Save metadata
# -----------------------------
metadata = {
    "model": "Random Forest Classifier",
    "dataset": "Iris Dataset",
    "features": feature_names,
    "classes": target_names.tolist(),
    "accuracy": float(accuracy),
    "test_size": 0.20,
    "random_state": 42
}

with open(
    models_dir / "model_meta.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)


print("\nModel files saved successfully!")