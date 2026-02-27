# OCI Model Images

OCI-compliant container images for machine learning models.

## Overview

This repository contains tools and configurations for building and managing OCI-compliant container images for ML models that can be used with KServe.

Currently supported ML frameworks:

- **scikit-learn** - `mlserver-sklearn`
- **XGBoost** - `mlserver-xgboost`
- **LightGBM** - `mlserver-lightgbm`
- **ONNX** - `mlserver-onnx` (CPU/GPU compatible)

## Getting Started

### Prerequisites

- Podman (or Docker)
- Access to a container registry (e.g., quay.io)
- Python 3.12+ (for training models)
- uv (for Python dependency management)

### Training Models

Train models using Iris dataset:

```bash
# Install dependencies first
make install-deps

# Train all models
make train-all

# Or train specific models
make train-sklearn
make train-xgboost
make train-lightgbm
make train-onnx
```

All trained models produce the expected output `[1, 1]` for the reference test data:

```python
[[6.8, 2.8, 4.8, 1.4], [6.0, 3.4, 4.5, 1.6]]
```

### Building Images

Build all model images:

```bash
make build-all
```

Build a specific model image:

```bash
make build-sklearn
make build-xgboost
make build-lightgbm
make build-onnx
```

### Customizing Registry and Versions

You can override the default registry and framework versions:

```bash
# Build with custom sklearn version
make SKLEARN_VERSION=1.9.0 build-sklearn

# Build all with custom registry
make REGISTRY=quay.io/your-username build-all

# Build specific version for each framework
make SKLEARN_VERSION=1.9.0 XGBOOST_VERSION=3.3.0 LIGHTGBM_VERSION=4.7.0 build-all
```

Default values:

- `REGISTRY`: `quay.io/jooholee`
- `SKLEARN_VERSION`: `1.8.0` (scikit-learn framework version)
- `XGBOOST_VERSION`: `3.2.0` (XGBoost framework version)
- `LIGHTGBM_VERSION`: `4.6.0` (LightGBM framework version)
- `ONNX_VERSION`: `1.20.1` (ONNX framework version)

### Pushing Images

Push all images to registry:

```bash
# Login to your registry first
podman login quay.io

# Push all images
make push-all
```

Push a specific image:

```bash
make push-sklearn
```

### Build and Push Everything

```bash
make all
```

### Available Make Targets

- `help` - Display help information
- `build-sklearn` - Build sklearn model image
- `build-xgboost` - Build xgboost model image
- `build-lightgbm` - Build lightgbm model image
- `build-all` - Build all model images
- `push-sklearn` - Push sklearn image to registry
- `push-xgboost` - Push xgboost image to registry
- `push-lightgbm` - Push lightgbm image to registry
- `push-all` - Push all images to registry
- `all` - Build and push all images
- `clean` - Remove local images

## Usage with KServe

Once the images are pushed to your registry, you can use them with KServe InferenceService:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sklearn-iris
spec:
  predictor:
    model:
      runtime: mlserver
      modelFormat:
        name: sklearn
      storageUri: oci://quay.io/jooholee/mlserver-sklearn:1.8.0
      # Or use latest tag: oci://quay.io/jooholee/mlserver-sklearn:latest
```

## Project Structure

```text
.
├── Dockerfile.sklearn      # Dockerfile for sklearn model
├── Dockerfile.xgboost      # Dockerfile for xgboost model
├── Dockerfile.lightgbm     # Dockerfile for lightgbm model
├── Dockerfile.onnx         # Dockerfile for ONNX model
├── Makefile                # Build and training automation
├── requirements.txt        # Python dependencies
├── train-sklearn.py        # sklearn model training script
├── train-xgboost.py        # XGBoost model training script
├── train-lightgbm.py       # LightGBM model training script
├── train-onnx.py           # ONNX model training script
├── README.md
└── models/
    └── mlserver/
        ├── sklearn/
        │   └── model.joblib    # sklearn model file
        ├── xgboost/
        │   └── model.bst       # XGBoost model file
        ├── lightgbm/
        │   ├── model.bst       # LightGBM model file (MLServer wellknown)
        │   └── model.txt       # LightGBM model file (alternative)
        └── onnx/
            └── model.onnx      # ONNX model file (CPU/GPU compatible)
```

## Image Details

All images are built with:

- **Base Image**: Red Hat UBI 10 Micro (latest)
- **Model Location**: `/models` directory
- **User**: Non-root user (UID 1001)
- **Format**: OCI-compliant
- **Optimization**: Squashed layers for minimal size

### Framework Versions and Tags

Each image is tagged with its framework version:

- **scikit-learn**: `quay.io/jooholee/mlserver-sklearn:1.8.0`
- **XGBoost**: `quay.io/jooholee/mlserver-xgboost:3.2.0`
- **LightGBM**: `quay.io/jooholee/mlserver-lightgbm:4.6.0`
- **ONNX**: `quay.io/jooholee/mlserver-onnx:1.20.1` (CPU/GPU compatible)

All images also have a `latest` tag pointing to the most recent build.

**Note on ONNX**: The ONNX model is framework-agnostic and can run on both CPU and GPU using `onnxruntime` (CPU) or `onnxruntime-gpu` (GPU) without changing the model file.

### OpenShift Compatibility

Images are configured for OpenShift security constraints:

- **File Ownership**: `1001:0` (user:group) - Group 0 (root) is required for OpenShift
- **Directory Permissions**: `555 (r-xr-xr-x)` - Read and execute for all users
- **File Permissions**: `444 (r--r--r--)` - Read for all users
- **Random UID Support**: OpenShift runs containers with random UID but maintains root group (GID 0), allowing access to group-owned files

These permission settings ensure model files are accessible regardless of the random UID assigned by OpenShift's security context constraints.

## License

TBD
