#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/mnt/nvme/hrm}"
IMAGE="${RRN_IMAGE:-hrm-rrn-train:cuda12}"
RUN_NAME="${RUN_NAME:-rrn-sudoku-1k}"
NGPU="${NGPU:-1}"

mkdir -p "$DATA_ROOT/checkpoints/$RUN_NAME"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -q -f docker/DockerfileRrnTrain -t "$IMAGE" docker >/dev/null
fi

docker run -d --gpus all --ipc=host --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e PYTHONPATH=/data/python-packages \
    -v "$PWD:/work" -w /work/rrn \
    -v "$DATA_ROOT/built:/data/built:ro" \
    -v "$DATA_ROOT/checkpoints:/data/checkpoints" \
    -v "$DATA_ROOT/python-packages:/data/python-packages:ro" \
    "$IMAGE" torchrun --nproc-per-node "$NGPU" train.py \
    --data /data/built/sudoku-extreme-1k-aug-1000 \
    --out "/data/checkpoints/$RUN_NAME" "$@"
