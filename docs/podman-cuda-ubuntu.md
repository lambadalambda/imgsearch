# Podman + CUDA (Ubuntu Container)

This document describes running `imgsearch` on a CUDA-capable Linux host with Podman, while keeping the app runtime in an Ubuntu-based image.

## Prerequisites

- Podman installed on the host.
- NVIDIA container runtime integration configured for Podman.
- GPU pass-through works with:

```bash
podman run --rm --gpus=all nvidia/cuda:12.8.1-runtime-ubuntu24.04 nvidia-smi
```

- `deps/llama.cpp` submodule initialized in your checkout:

```bash
git submodule update --init --recursive deps/llama.cpp
```

## Build The CUDA Image

From the repository root:

```bash
podman build \
  -f Containerfile.cuda \
  -t imgsearch:cuda \
  .
```

Optional GPU architecture tuning during build:

```bash
podman build \
  -f Containerfile.cuda \
  -t imgsearch:cuda \
  --build-arg CUDA_DOCKER_ARCH=89 \
  .
```

(`CUDA_DOCKER_ARCH=default` is the default and builds for a broader target set.)

## Run With Podman

Create persistent host directories first:

```bash
mkdir -p "$HOME/imgsearch-data" "$HOME/imgsearch-models"
```

Run:

```bash
podman run -d \
  --name imgsearch \
  --replace \
  --gpus=all \
  -p 8080:8080 \
  -v "$HOME/imgsearch-data:/data" \
  -v "$HOME/imgsearch-models:/models" \
  imgsearch:cuda
```

Then open:

- `http://<host-ip>:8080/`

The first run downloads default embedding + annotator models into `/models` (your mounted `~/imgsearch-models` directory).

## Useful Commands

Follow logs:

```bash
podman logs -f imgsearch
```

Stop:

```bash
podman stop imgsearch
```

Remove container:

```bash
podman rm -f imgsearch
```

## Runtime Overrides

Disable annotations:

```bash
podman run -d \
  --name imgsearch \
  --replace \
  --gpus=all \
  -p 8080:8080 \
  -v "$HOME/imgsearch-data:/data" \
  -v "$HOME/imgsearch-models:/models" \
  -e IMGSEARCH_ENABLE_ANNOTATIONS=false \
  imgsearch:cuda
```

Run split worker-only process:

```bash
podman run --rm \
  --name imgsearch-worker \
  --gpus=all \
  -v "$HOME/imgsearch-data:/data" \
  -v "$HOME/imgsearch-models:/models" \
  -e IMGSEARCH_MODE=worker \
  imgsearch:cuda
```

Pass direct app flags (appended to entrypoint command):

```bash
podman run --rm \
  --gpus=all \
  -p 8080:8080 \
  -v "$HOME/imgsearch-data:/data" \
  -v "$HOME/imgsearch-models:/models" \
  imgsearch:cuda \
  -llama-native-gpu-layers 99 \
  -worker-batch-size 2
```
