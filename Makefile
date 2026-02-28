# OCI Model Images Build Automation
# Variables can be overridden from command line:
# make REGISTRY=quay.io/custom SKLEARN_VERSION=1.9.0 build-sklearn

REGISTRY ?= quay.io/jooholee
SKLEARN_VERSION ?= 1.8.0
XGBOOST_VERSION ?= 3.2.0
LIGHTGBM_VERSION ?= 4.6.0
ONNX_VERSION ?= 1.20.1

IMAGE_SKLEARN = $(REGISTRY)/mlserver-sklearn:$(SKLEARN_VERSION)
IMAGE_SKLEARN_LATEST = $(REGISTRY)/mlserver-sklearn:latest
IMAGE_XGBOOST = $(REGISTRY)/mlserver-xgboost:$(XGBOOST_VERSION)
IMAGE_XGBOOST_LATEST = $(REGISTRY)/mlserver-xgboost:latest
IMAGE_LIGHTGBM = $(REGISTRY)/mlserver-lightgbm:$(LIGHTGBM_VERSION)
IMAGE_LIGHTGBM_LATEST = $(REGISTRY)/mlserver-lightgbm:latest
IMAGE_ONNX = $(REGISTRY)/mlserver-onnx:$(ONNX_VERSION)
IMAGE_ONNX_LATEST = $(REGISTRY)/mlserver-onnx:latest

.PHONY: help train-sklearn train-xgboost train-lightgbm train-onnx train-all \
        test-sklearn test-xgboost test-lightgbm test-onnx test-all \
        build-sklearn build-xgboost build-lightgbm build-onnx build-all \
        push-sklearn push-xgboost push-lightgbm push-onnx push-all \
        all clean install-deps

help:
	@echo "OCI Model Images - Build Automation"
	@echo ""
	@echo "Variables:"
	@echo "  REGISTRY         = $(REGISTRY)"
	@echo "  SKLEARN_VERSION  = $(SKLEARN_VERSION)"
	@echo "  XGBOOST_VERSION  = $(XGBOOST_VERSION)"
	@echo "  LIGHTGBM_VERSION = $(LIGHTGBM_VERSION)"
	@echo "  ONNX_VERSION     = $(ONNX_VERSION)"
	@echo ""
	@echo "Images:"
	@echo "  sklearn   = $(IMAGE_SKLEARN)"
	@echo "  xgboost   = $(IMAGE_XGBOOST)"
	@echo "  lightgbm  = $(IMAGE_LIGHTGBM)"
	@echo "  onnx      = $(IMAGE_ONNX)"
	@echo ""
	@echo "Targets:"
	@echo "  install-deps    - Install Python dependencies using uv"
	@echo "  train-sklearn   - Train sklearn model on Iris dataset"
	@echo "  train-xgboost   - Train xgboost model on Iris dataset"
	@echo "  train-lightgbm  - Train lightgbm model on Iris dataset"
	@echo "  train-onnx      - Train and convert model to ONNX format (CPU/GPU compatible)"
	@echo "  train-all       - Train all models"
	@echo ""
	@echo "  test-sklearn    - Test sklearn model inference"
	@echo "  test-xgboost    - Test xgboost model inference"
	@echo "  test-lightgbm   - Test lightgbm model inference"
	@echo "  test-onnx       - Test ONNX model inference"
	@echo "  test-all        - Test all model inferences"
	@echo ""
	@echo "  build-sklearn   - Build sklearn model image"
	@echo "  build-xgboost   - Build xgboost model image"
	@echo "  build-lightgbm  - Build lightgbm model image"
	@echo "  build-onnx      - Build ONNX model image"
	@echo "  build-all       - Build all model images"
	@echo ""
	@echo "  push-sklearn    - Push sklearn image to registry"
	@echo "  push-xgboost    - Push xgboost image to registry"
	@echo "  push-lightgbm   - Push lightgbm image to registry"
	@echo "  push-onnx       - Push ONNX image to registry"
	@echo "  push-all        - Push all images to registry"
	@echo ""
	@echo "  all             - Build and push all images"
	@echo "  clean           - Remove local images"
	@echo ""
	@echo "Examples:"
	@echo "  make build-all"
	@echo "  make REGISTRY=quay.io/custom SKLEARN_VERSION=1.9.0 build-sklearn"
	@echo "  make push-all"

# Dependencies
install-deps:
	@echo "Installing Python dependencies with uv..."
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		uv venv --python 3.12; \
	fi
	@echo "Installing packages from requirements.txt..."
	@source .venv/bin/activate && uv pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Training targets
train-sklearn:
	@echo "Training sklearn model..."
	PYTHONPATH=.venv/lib/python3.12/site-packages python3.12 train-sklearn.py
	@echo "✓ sklearn model trained"

train-xgboost:
	@echo "Training xgboost model..."
	PYTHONPATH=.venv/lib/python3.12/site-packages python3.12 train-xgboost.py
	@echo "✓ xgboost model trained"

train-lightgbm:
	@echo "Training lightgbm model..."
	PYTHONPATH=.venv/lib/python3.12/site-packages python3.12 train-lightgbm.py
	@echo "✓ lightgbm model trained"

train-onnx:
	@echo "Training and converting model to ONNX format..."
	PYTHONPATH=.venv/lib/python3.12/site-packages python3.12 train-onnx.py
	@echo "✓ ONNX model trained and converted"

train-all: train-sklearn train-xgboost train-lightgbm train-onnx
	@echo ""
	@echo "✓ All models trained successfully"

# Test targets
test-sklearn:
	@echo "Testing sklearn model inference..."
	PYTHONPATH=.venv/lib/python3.12/site-packages python3.12 test-sklearn.py

test-xgboost:
	@echo "Testing xgboost model inference..."
	PYTHONPATH=.venv/lib/python3.12/site-packages python3.12 test-xgboost.py

test-lightgbm:
	@echo "Testing lightgbm model inference..."
	PYTHONPATH=.venv/lib/python3.12/site-packages python3.12 test-lightgbm.py

test-onnx:
	@echo "Testing ONNX model inference..."
	PYTHONPATH=.venv/lib/python3.12/site-packages python3.12 test-onnx.py

test-all: test-sklearn test-xgboost test-lightgbm test-onnx
	@echo ""
	@echo "✓ All model inference tests passed"

# Build targets
build-sklearn:
	@echo "Building sklearn model image..."
	podman build --format=oci --squash -f Dockerfile.sklearn -t $(IMAGE_SKLEARN) -t $(IMAGE_SKLEARN_LATEST) .
	@echo "✓ Built: $(IMAGE_SKLEARN)"
	@echo "✓ Tagged: $(IMAGE_SKLEARN_LATEST)"

build-xgboost:
	@echo "Building xgboost model image..."
	podman build --format=oci --squash -f Dockerfile.xgboost -t $(IMAGE_XGBOOST) -t $(IMAGE_XGBOOST_LATEST) .
	@echo "✓ Built: $(IMAGE_XGBOOST)"
	@echo "✓ Tagged: $(IMAGE_XGBOOST_LATEST)"

build-lightgbm:
	@echo "Building lightgbm model image..."
	podman build --format=oci --squash -f Dockerfile.lightgbm -t $(IMAGE_LIGHTGBM) -t $(IMAGE_LIGHTGBM_LATEST) .
	@echo "✓ Built: $(IMAGE_LIGHTGBM)"
	@echo "✓ Tagged: $(IMAGE_LIGHTGBM_LATEST)"

build-onnx:
	@echo "Building ONNX model image..."
	podman build --format=oci --squash -f Dockerfile.onnx -t $(IMAGE_ONNX) -t $(IMAGE_ONNX_LATEST) .
	@echo "✓ Built: $(IMAGE_ONNX)"
	@echo "✓ Tagged: $(IMAGE_ONNX_LATEST)"

build-all: build-sklearn build-xgboost build-lightgbm build-onnx
	@echo ""
	@echo "✓ All images built successfully"

# Push targets
push-sklearn:
	@echo "Pushing sklearn image to registry..."
	podman push $(IMAGE_SKLEARN)
	podman push $(IMAGE_SKLEARN_LATEST)
	@echo "✓ Pushed: $(IMAGE_SKLEARN)"
	@echo "✓ Pushed: $(IMAGE_SKLEARN_LATEST)"

push-xgboost:
	@echo "Pushing xgboost image to registry..."
	podman push $(IMAGE_XGBOOST)
	podman push $(IMAGE_XGBOOST_LATEST)
	@echo "✓ Pushed: $(IMAGE_XGBOOST)"
	@echo "✓ Pushed: $(IMAGE_XGBOOST_LATEST)"

push-lightgbm:
	@echo "Pushing lightgbm image to registry..."
	podman push $(IMAGE_LIGHTGBM)
	podman push $(IMAGE_LIGHTGBM_LATEST)
	@echo "✓ Pushed: $(IMAGE_LIGHTGBM)"
	@echo "✓ Pushed: $(IMAGE_LIGHTGBM_LATEST)"

push-onnx:
	@echo "Pushing ONNX image to registry..."
	podman push $(IMAGE_ONNX)
	podman push $(IMAGE_ONNX_LATEST)
	@echo "✓ Pushed: $(IMAGE_ONNX)"
	@echo "✓ Pushed: $(IMAGE_ONNX_LATEST)"

push-all: push-sklearn push-xgboost push-lightgbm push-onnx
	@echo ""
	@echo "✓ All images pushed successfully"

# Combined target
all: build-all push-all
	@echo ""
	@echo "✓ Build and push completed for all images"

# Cleanup
clean:
	@echo "Removing local images..."
	-podman rmi $(IMAGE_SKLEARN) $(IMAGE_SKLEARN_LATEST) 2>/dev/null || true
	-podman rmi $(IMAGE_XGBOOST) $(IMAGE_XGBOOST_LATEST) 2>/dev/null || true
	-podman rmi $(IMAGE_LIGHTGBM) $(IMAGE_LIGHTGBM_LATEST) 2>/dev/null || true
	-podman rmi $(IMAGE_ONNX) $(IMAGE_ONNX_LATEST) 2>/dev/null || true
	@echo "✓ Cleanup complete"
