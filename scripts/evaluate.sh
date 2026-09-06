#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/mnt/nvme/hrm}"
IMAGE="${TRAIN_IMAGE:-hrm-train:cuda12}"
NGPU="${NGPU:-1}"

docker run --rm --init --gpus all --ipc=host --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e DISABLE_COMPILE -e COMPILE_MODE -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    -e TRITON_CACHE_DIR=/tmp/triton \
    -e METRICS_FLUSH -e PYTHONPATH=/data/python-packages -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
    -v "$PWD:/work" -w /work \
    -v "$DATA_ROOT/built:/data/built:ro" \
    -v "$DATA_ROOT/checkpoints:/data/checkpoints" \
    "$IMAGE" torchrun --nproc-per-node "$NGPU" evaluate.py "$@"
