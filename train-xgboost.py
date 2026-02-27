#!/usr/bin/env python3
"""
Train XGBoost model on Iris dataset and save as model.bst
Expected output for test data [6.8, 2.8, 4.8, 1.4] and [6.0, 3.4, 4.5, 1.6]: [1, 1]
"""

import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
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

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train XGBoost classifier
print("\nTraining XGBoost classifier...")
params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 3,
    'eta': 0.3,
    'seed': 42
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    verbose_eval=False
)

# Evaluate
train_pred = model.predict(dtrain)
test_pred = model.predict(dtest)
train_accuracy = np.mean(train_pred == y_train)
test_accuracy = np.mean(test_pred == y_test)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Test with expected data
test_data = np.array([
    [6.8, 2.8, 4.8, 1.4],  # Expected: 1 (Iris-versicolor)
    [6.0, 3.4, 4.5, 1.6]   # Expected: 1 (Iris-versicolor)
])

dtest_ref = xgb.DMatrix(test_data)
predictions = model.predict(dtest_ref)
print(f"\nTest predictions for reference data:")
print(f"Input: {test_data.tolist()}")
print(f"Predictions: {predictions.astype(int).tolist()}")
print(f"Expected: [1, 1]")

# Save model
output_path = "models/mlserver/xgboost/model.bst"
print(f"\nSaving model to {output_path}...")
model.save_model(output_path)
print("✓ Model saved successfully")

# Verify saved model
print("\nVerifying saved model...")
loaded_model = xgb.Booster()
loaded_model.load_model(output_path)
verify_predictions = loaded_model.predict(dtest_ref)
print(f"Loaded model predictions: {verify_predictions.astype(int).tolist()}")
assert np.array_equal(predictions, verify_predictions), "Model verification failed!"
print("✓ Model verification passed")
