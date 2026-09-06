#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/mnt/nvme/hrm}"
IMAGE="${TRAIN_IMAGE:-hrm-train:cuda12}"
DOCKERFILE="${TRAIN_DOCKERFILE:-docker/DockerfileTrain}"
NGPU="${NGPU:-1}"
RUN_NAME="${RUN_NAME:-run}"

mkdir -p "$DATA_ROOT/checkpoints" "$DATA_ROOT/torch-cache" "$DATA_ROOT/triton-cache"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -q -f "$DOCKERFILE" -t "$IMAGE" docker
docker run --rm --init --gpus all --ipc=host --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e DISABLE_COMPILE -e COMPILE_MODE -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e METRICS_FLUSH -e PYTHONPATH=/data/python-packages -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
    -e TORCHINDUCTOR_CACHE_DIR=/data/torch-cache \
    -e TRITON_CACHE_DIR=/data/triton-cache \
    -v "$PWD:/work" -w /work \
    -v "$DATA_ROOT/built:/data/built:ro" \
    -v "$DATA_ROOT/checkpoints:/data/checkpoints" \
    -v "$DATA_ROOT/torch-cache:/data/torch-cache" \
    -v "$DATA_ROOT/python-packages:/data/python-packages:ro" \
    -v "$DATA_ROOT/triton-cache:/data/triton-cache" \
    "$IMAGE" torchrun --nproc-per-node "$NGPU" pretrain.py \
    checkpoint_path="/data/checkpoints/$RUN_NAME" "$@"
