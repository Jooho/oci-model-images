#!/usr/bin/env python3
"""
Train PyTorch classifier on Iris dataset and export to ONNX format
ONNX models can run on both CPU and GPU with onnxruntime
Expected output for test data [6.8, 2.8, 4.8, 1.4] and [6.0, 3.4, 4.5, 1.6]: [1, 1]
"""

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as rt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset
print("Loading Iris dataset...")
iris = load_iris()
X = iris['data'].astype(np.float32)
y = iris['target'].astype(np.int64)

print(f"Dataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")
print(f"Feature names: {iris.feature_names}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)


# Define a simple neural network classifier
class IrisClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# Create and train the model
print("\nTraining PyTorch classifier...")
torch.manual_seed(42)
model = IrisClassifier(input_dim=4, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"  Epoch [{epoch+1}/200], Loss: {loss.item():.4f}")

# Evaluate PyTorch model
model.eval()
with torch.no_grad():
    train_pred = model(X_train_tensor).argmax(dim=1)
    test_pred = model(X_test_tensor).argmax(dim=1)
    train_accuracy = (train_pred == y_train_tensor).float().mean()
    test_accuracy = (test_pred == y_test_tensor).float().mean()

print(f"\nTraining accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Test with expected data
test_data = np.array([
    [6.8, 2.8, 4.8, 1.4],  # Expected: 1 (Iris-versicolor)
    [6.0, 3.4, 4.5, 1.6]   # Expected: 1 (Iris-versicolor)
], dtype=np.float32)

with torch.no_grad():
    predictions_torch = model(torch.FloatTensor(test_data)).argmax(dim=1).numpy()
print(f"\nTest predictions from PyTorch model:")
print(f"Input: {test_data.tolist()}")
print(f"Predictions: {predictions_torch.tolist()}")
print(f"Expected: [1, 1]")

# Export to ONNX
print("\nExporting model to ONNX format...")
dummy_input = torch.randn(1, 4)
output_path = "models/mlserver/onnx/model.onnx"

torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
print("✓ ONNX model exported successfully")

# Verify ONNX model with onnxruntime (works on both CPU and GPU)
print("\nVerifying ONNX model with onnxruntime...")
sess = rt.InferenceSession(output_path, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

onnx_output = sess.run([output_name], {input_name: test_data})[0]
predictions_onnx = np.argmax(onnx_output, axis=1)
print(f"ONNX model predictions: {predictions_onnx.tolist()}")
assert np.array_equal(predictions_torch, predictions_onnx), "ONNX model verification failed!"
print("✓ ONNX model verification passed")

print("\nNote: This ONNX model can run on both CPU and GPU:")
print("  - CPU: onnxruntime with CPUExecutionProvider")
print("  - GPU: onnxruntime-gpu with CUDAExecutionProvider")
