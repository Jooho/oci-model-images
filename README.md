# OCI Model Images

OCI-compliant container images for machine learning models.

## Overview

This repository contains tools and configurations for building and managing OCI-compliant container images for ML models that can be used with KServe.

Currently supported ML frameworks:

- **scikit-learn** - `mlserver-sklearn`
- **XGBoost** - `mlserver-xgboost`
- **LightGBM** - `mlserver-lightgbm`

## Getting Started

### Prerequisites

- Podman (or Docker)
- Access to a container registry (e.g., quay.io)

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
```

### Customizing Registry and Version

You can override the default registry and version:

```bash
make REGISTRY=quay.io/your-username VERSION=v2.0.0 build-all
```

Default values:

- `REGISTRY`: `quay.io/jooholee`
- `VERSION`: `v1.0.0`

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
      storageUri: oci://quay.io/jooholee/mlserver-sklearn:v1.0.0
```

## Project Structure

```text
.
├── Dockerfile.sklearn      # Dockerfile for sklearn model
├── Dockerfile.xgboost      # Dockerfile for xgboost model
├── Dockerfile.lightgbm     # Dockerfile for lightgbm model
├── Makefile                # Build automation
├── README.md
└── models/
    └── mlserver/
        ├── sklearn/
        │   └── model.joblib
        ├── xgboost/
        │   └── model.bst
        └── lightgbm/
            └── model.bst
```

## Image Details

All images are built with:

- **Base Image**: Red Hat UBI 10 Micro (latest)
- **Model Location**: `/models` directory
- **User**: Non-root user (UID 1001)
- **Format**: OCI-compliant
- **Optimization**: Squashed layers for minimal size

### OpenShift Compatibility

Images are configured for OpenShift security constraints:

- **File Ownership**: `1001:0` (user:group) - Group 0 (root) is required for OpenShift
- **Directory Permissions**: `555 (r-xr-xr-x)` - Read and execute for all users
- **File Permissions**: `444 (r--r--r--)` - Read for all users
- **Random UID Support**: OpenShift runs containers with random UID but maintains root group (GID 0), allowing access to group-owned files

These permission settings ensure model files are accessible regardless of the random UID assigned by OpenShift's security context constraints.

## License

TBD
