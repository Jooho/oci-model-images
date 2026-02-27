#!/usr/bin/env python3
"""
Train LightGBM model on Iris dataset and save in both formats (model.txt and model.bst)
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

# Save model in both formats for flexibility
output_path_txt = "models/mlserver/lightgbm/model.txt"
output_path_bst = "models/mlserver/lightgbm/model.bst"

print(f"\nSaving model to {output_path_txt}...")
model.save_model(output_path_txt)
print("✓ Model saved as model.txt")

print(f"Saving model to {output_path_bst}...")
model.save_model(output_path_bst)
print("✓ Model saved as model.bst (MLServer wellknown filename)")

# Verify both saved models
print("\nVerifying saved models...")
loaded_model_txt = lgb.Booster(model_file=output_path_txt)
loaded_model_bst = lgb.Booster(model_file=output_path_bst)

verify_predictions_txt = np.argmax(loaded_model_txt.predict(test_data_ref), axis=1)
verify_predictions_bst = np.argmax(loaded_model_bst.predict(test_data_ref), axis=1)

print(f"model.txt predictions: {verify_predictions_txt.tolist()}")
print(f"model.bst predictions: {verify_predictions_bst.tolist()}")

assert np.array_equal(predictions, verify_predictions_txt), "model.txt verification failed!"
assert np.array_equal(predictions, verify_predictions_bst), "model.bst verification failed!"
print("✓ Both models verification passed")
