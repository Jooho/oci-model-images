#!/usr/bin/env python3
"""
Train sklearn model on Iris dataset and save as model.joblib
Expected output for test data [6.8, 2.8, 4.8, 1.4] and [6.0, 3.4, 4.5, 1.6]: [1, 1]
"""

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load iris dataset
print("Loading Iris dataset...")
iris = load_iris()
X = iris['data']
y = iris['target']

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")
print(f"Feature names: {iris.feature_names}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest classifier
print("\nTraining Random Forest classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# Test with expected data
test_data = np.array([
    [6.8, 2.8, 4.8, 1.4],  # Expected: 1 (Iris-versicolor)
    [6.0, 3.4, 4.5, 1.6]   # Expected: 1 (Iris-versicolor)
])

predictions = model.predict(test_data)
print(f"\nTest predictions for reference data:")
print(f"Input: {test_data.tolist()}")
print(f"Predictions: {predictions.tolist()}")
print(f"Expected: [1, 1]")

# Save model
output_path = "models/mlserver/sklearn/model.joblib"
print(f"\nSaving model to {output_path}...")
joblib.dump(model, output_path)
print("✓ Model saved successfully")

# Verify saved model
print("\nVerifying saved model...")
loaded_model = joblib.load(output_path)
verify_predictions = loaded_model.predict(test_data)
print(f"Loaded model predictions: {verify_predictions.tolist()}")
assert np.array_equal(predictions, verify_predictions), "Model verification failed!"
print("✓ Model verification passed")
