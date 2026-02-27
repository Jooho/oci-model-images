#!/usr/bin/env python3
"""
Train sklearn model on Iris dataset and convert to ONNX format
ONNX models can run on both CPU and GPU with onnxruntime
Expected output for test data [6.8, 2.8, 4.8, 1.4] and [6.0, 3.4, 4.5, 1.6]: [1, 1]
"""

import numpy as np
import onnxruntime as rt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

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

# Train RandomForest classifier
print("\nTraining RandomForest classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate sklearn model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Test with expected data (sklearn model)
test_data = np.array([
    [6.8, 2.8, 4.8, 1.4],  # Expected: 1 (Iris-versicolor)
    [6.0, 3.4, 4.5, 1.6]   # Expected: 1 (Iris-versicolor)
], dtype=np.float32)

predictions_sklearn = model.predict(test_data)
print(f"\nTest predictions from sklearn model:")
print(f"Input: {test_data.tolist()}")
print(f"Predictions: {predictions_sklearn.tolist()}")
print(f"Expected: [1, 1]")

# Convert to ONNX
print("\nConverting model to ONNX format...")
initial_type = [('float_input', FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    target_opset=12  # Compatible with most ONNX runtimes
)

# Save ONNX model
output_path = "models/mlserver/onnx/model.onnx"
print(f"\nSaving ONNX model to {output_path}...")
with open(output_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print("✓ ONNX model saved successfully")

# Verify ONNX model with onnxruntime (works on both CPU and GPU)
print("\nVerifying ONNX model with onnxruntime...")
sess = rt.InferenceSession(output_path, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

predictions_onnx = sess.run([label_name], {input_name: test_data})[0]
print(f"ONNX model predictions: {predictions_onnx.tolist()}")
assert np.array_equal(predictions_sklearn, predictions_onnx), "ONNX model verification failed!"
print("✓ ONNX model verification passed")

print("\nNote: This ONNX model can run on both CPU and GPU:")
print("  - CPU: onnxruntime with CPUExecutionProvider")
print("  - GPU: onnxruntime-gpu with CUDAExecutionProvider")
