#!/usr/bin/env python3
"""
Train LightGBM model on Iris dataset and save as model.txt
Expected output for test data [6.8, 2.8, 4.8, 1.4] and [6.0, 3.4, 4.5, 1.6]: [1, 1]
"""

import numpy as np
import lightgbm as lgb
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

# Create Dataset for LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Train LightGBM classifier
print("\nTraining LightGBM classifier...")
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'seed': 42,
    'verbose': -1
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'test'],
    callbacks=[lgb.log_evaluation(period=0)]
)

# Evaluate
train_pred = np.argmax(model.predict(X_train), axis=1)
test_pred = np.argmax(model.predict(X_test), axis=1)
train_accuracy = np.mean(train_pred == y_train)
test_accuracy = np.mean(test_pred == y_test)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Test with expected data
test_data_ref = np.array([
    [6.8, 2.8, 4.8, 1.4],  # Expected: 1 (Iris-versicolor)
    [6.0, 3.4, 4.5, 1.6]   # Expected: 1 (Iris-versicolor)
])

predictions_proba = model.predict(test_data_ref)
predictions = np.argmax(predictions_proba, axis=1)
print(f"\nTest predictions for reference data:")
print(f"Input: {test_data_ref.tolist()}")
print(f"Predictions: {predictions.tolist()}")
print(f"Expected: [1, 1]")

# Save model
output_path = "models/mlserver/lightgbm/model.txt"
print(f"\nSaving model to {output_path}...")
model.save_model(output_path)
print("✓ Model saved successfully")

# Verify saved model
print("\nVerifying saved model...")
loaded_model = lgb.Booster(model_file=output_path)
verify_predictions_proba = loaded_model.predict(test_data_ref)
verify_predictions = np.argmax(verify_predictions_proba, axis=1)
print(f"Loaded model predictions: {verify_predictions.tolist()}")
assert np.array_equal(predictions, verify_predictions), "Model verification failed!"
print("✓ Model verification passed")
