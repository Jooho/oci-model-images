#!/usr/bin/env python3
"""
Test LightGBM model inference
Validates that the trained model can be loaded and produces expected predictions
Tests both model.bst and model.txt formats
"""

import numpy as np
import lightgbm as lgb

# Test data - same as training reference data
test_data = np.array([
    [6.8, 2.8, 4.8, 1.4],  # Expected: 1 (Iris-versicolor)
    [6.0, 3.4, 4.5, 1.6]   # Expected: 1 (Iris-versicolor)
])
expected_output = [1, 1]

print("=" * 60)
print("Testing LightGBM Model Inference")
print("=" * 60)

# Test model.bst (MLServer wellknown format)
model_path_bst = "models/mlserver/lightgbm/model.bst"
print(f"\n[1/2] Testing model.bst")
print(f"Loading model from {model_path_bst}...")
model_bst = lgb.Booster(model_file=model_path_bst)
print("✓ Model loaded successfully")

print(f"\nPerforming inference on test data:")
print(f"Input shape: {test_data.shape}")
print(f"Input data:\n{test_data}")

predictions_bst = np.argmax(model_bst.predict(test_data), axis=1)
print(f"\nPredictions: {predictions_bst.tolist()}")
print(f"Expected:    {expected_output}")

bst_passed = np.array_equal(predictions_bst, expected_output)
if bst_passed:
    print("✓ model.bst inference validation PASSED")
else:
    print(f"✗ model.bst inference validation FAILED")

# Test model.txt (alternative format)
model_path_txt = "models/mlserver/lightgbm/model.txt"
print(f"\n[2/2] Testing model.txt")
print(f"Loading model from {model_path_txt}...")
model_txt = lgb.Booster(model_file=model_path_txt)
print("✓ Model loaded successfully")

predictions_txt = np.argmax(model_txt.predict(test_data), axis=1)
print(f"\nPredictions: {predictions_txt.tolist()}")
print(f"Expected:    {expected_output}")

txt_passed = np.array_equal(predictions_txt, expected_output)
if txt_passed:
    print("✓ model.txt inference validation PASSED")
else:
    print(f"✗ model.txt inference validation FAILED")

# Final result
print("\n" + "=" * 60)
if bst_passed and txt_passed:
    print("✓ ALL TESTS PASSED")
    print("Both model formats are working correctly!")
    exit(0)
else:
    print("✗ SOME TESTS FAILED")
    if not bst_passed:
        print("  - model.bst validation failed")
    if not txt_passed:
        print("  - model.txt validation failed")
    exit(1)
