#!/usr/bin/env python3
"""
Test sklearn model inference
Validates that the trained model can be loaded and produces expected predictions
"""

import numpy as np
import joblib

# Test data - same as training reference data
test_data = np.array([
    [6.8, 2.8, 4.8, 1.4],  # Expected: 1 (Iris-versicolor)
    [6.0, 3.4, 4.5, 1.6]   # Expected: 1 (Iris-versicolor)
])
expected_output = [1, 1]

print("=" * 60)
print("Testing sklearn Model Inference")
print("=" * 60)

# Load model
model_path = "models/mlserver/sklearn/model.joblib"
print(f"\nLoading model from {model_path}...")
model = joblib.load(model_path)
print("✓ Model loaded successfully")

# Perform inference
print(f"\nPerforming inference on test data:")
print(f"Input shape: {test_data.shape}")
print(f"Input data:\n{test_data}")

predictions = model.predict(test_data)
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
