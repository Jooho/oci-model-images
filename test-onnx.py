#!/usr/bin/env python3
"""
Test ONNX model inference
Validates that the trained model can be loaded and produces expected predictions
Tests CPU execution (GPU-compatible model)
"""

import numpy as np
import onnxruntime as rt

# Test data - same as training reference data
test_data = np.array([
    [6.8, 2.8, 4.8, 1.4],  # Expected: 1 (Iris-versicolor)
    [6.0, 3.4, 4.5, 1.6]   # Expected: 1 (Iris-versicolor)
], dtype=np.float32)
expected_output = [1, 1]

print("=" * 60)
print("Testing ONNX Model Inference")
print("=" * 60)

# Load model
model_path = "models/mlserver/onnx/model.onnx"
print(f"\nLoading model from {model_path}...")
session = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
print("✓ Model loaded successfully")
print(f"Execution providers: {session.get_providers()}")

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"Input name: {input_name}")
print(f"Output name: {output_name}")

# Perform inference
print(f"\nPerforming inference on test data:")
print(f"Input shape: {test_data.shape}")
print(f"Input data:\n{test_data}")

predictions = session.run([output_name], {input_name: test_data})[0]
print(f"\nPredictions: {predictions.tolist()}")
print(f"Expected:    {expected_output}")

# Validate results
if np.array_equal(predictions, expected_output):
    print("\n✓ Inference validation PASSED")
    print("Model is working correctly!")
    print("\nNote: This model is GPU-compatible and can run with:")
    print("  - CPUExecutionProvider (tested)")
    print("  - CUDAExecutionProvider (requires onnxruntime-gpu)")
    exit(0)
else:
    print(f"\n✗ Inference validation FAILED")
    print(f"Expected {expected_output}, got {predictions.tolist()}")
    exit(1)
