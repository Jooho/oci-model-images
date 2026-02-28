#!/usr/bin/env python3
"""
Test XGBoost model inference
Validates that the trained model can be loaded and produces expected predictions
"""

import numpy as np
import xgboost as xgb

# Test data - same as training reference data
test_data = np.array([
    [6.8, 2.8, 4.8, 1.4],  # Expected: 1 (Iris-versicolor)
    [6.0, 3.4, 4.5, 1.6]   # Expected: 1 (Iris-versicolor)
], dtype=np.float32)
expected_output = [1, 1]

print("=" * 60)
print("Testing XGBoost Model Inference")
print("=" * 60)

# Load model
model_path = "models/mlserver/xgboost/model.bst"
print(f"\nLoading model from {model_path}...")
model = xgb.Booster()
model.load_model(model_path)
print("✓ Model loaded successfully")

# Perform inference
print(f"\nPerforming inference on test data:")
print(f"Input shape: {test_data.shape}")
print(f"Input data:\n{test_data}")

dtest = xgb.DMatrix(test_data)
predictions = model.predict(dtest).astype(int)
print(f"\nPredictions: {predictions.tolist()}")
print(f"Expected:    {expected_output}")

# Validate results
if np.array_equal(predictions, expected_output):
    print("\n✓ Inference validation PASSED")
    print("Model is working correctly!")
    exit(0)
else:
    print(f"\n✗ Inference validation FAILED")
    print(f"Expected {expected_output}, got {predictions.tolist()}")
    exit(1)
