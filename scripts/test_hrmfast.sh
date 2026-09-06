#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/mnt/nvme/hrm}"
IMAGE="${TRAIN_IMAGE:-hrm-train:cuda12}"

docker run --rm --init --gpus all --ipc=host --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e OMP_NUM_THREADS=8 \
    -e PYTHONPATH=/data/python-packages \
    -v "$PWD:/work" -w /work \
    -v "$DATA_ROOT/python-packages:/data/python-packages:ro" \
    "$IMAGE" sh -c "python -u hrmfast/tests/test_swiglu.py && python -u hrmfast/tests/test_fused_all.py && python -u hrmfast/tests/test_attn.py"
