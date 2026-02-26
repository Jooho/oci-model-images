# OCI Model Images Build Automation
# Variables can be overridden from command line:
# make REGISTRY=quay.io/custom VERSION=v2.0.0 build-all

REGISTRY ?= quay.io/jooholee
VERSION ?= v1.0.0

IMAGE_SKLEARN = $(REGISTRY)/mlserver-sklearn:$(VERSION)
IMAGE_SKLEARN_LATEST = $(REGISTRY)/mlserver-sklearn:latest
IMAGE_XGBOOST = $(REGISTRY)/mlserver-xgboost:$(VERSION)
IMAGE_XGBOOST_LATEST = $(REGISTRY)/mlserver-xgboost:latest
IMAGE_LIGHTGBM = $(REGISTRY)/mlserver-lightgbm:$(VERSION)
IMAGE_LIGHTGBM_LATEST = $(REGISTRY)/mlserver-lightgbm:latest

.PHONY: help build-sklearn build-xgboost build-lightgbm build-all \
        push-sklearn push-xgboost push-lightgbm push-all \
        all clean

help:
	@echo "OCI Model Images - Build Automation"
	@echo ""
	@echo "Variables:"
	@echo "  REGISTRY  = $(REGISTRY)"
	@echo "  VERSION   = $(VERSION)"
	@echo ""
	@echo "Images:"
	@echo "  sklearn   = $(IMAGE_SKLEARN)"
	@echo "  xgboost   = $(IMAGE_XGBOOST)"
	@echo "  lightgbm  = $(IMAGE_LIGHTGBM)"
	@echo ""
	@echo "Targets:"
	@echo "  build-sklearn   - Build sklearn model image"
	@echo "  build-xgboost   - Build xgboost model image"
	@echo "  build-lightgbm  - Build lightgbm model image"
	@echo "  build-all       - Build all model images"
	@echo ""
	@echo "  push-sklearn    - Push sklearn image to registry"
	@echo "  push-xgboost    - Push xgboost image to registry"
	@echo "  push-lightgbm   - Push lightgbm image to registry"
	@echo "  push-all        - Push all images to registry"
	@echo ""
	@echo "  all             - Build and push all images"
	@echo "  clean           - Remove local images"
	@echo ""
	@echo "Examples:"
	@echo "  make build-all"
	@echo "  make REGISTRY=quay.io/custom VERSION=v2.0.0 all"

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

build-all: build-sklearn build-xgboost build-lightgbm
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

push-all: push-sklearn push-xgboost push-lightgbm
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
	@echo "✓ Cleanup complete"
