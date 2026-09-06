#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/mnt/nvme/hrm}"
IMAGE="${TRAIN_IMAGE:-hrm-train:cuda12}"

mkdir -p "$DATA_ROOT/python-packages"
docker run --rm --init --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$PWD:/work" -w /work/hrmfast \
    -v "$DATA_ROOT/python-packages:/data/python-packages" \
    "$IMAGE" sh -c "rm -rf /data/python-packages/hrmfast /data/python-packages/hrmfast_backend*.so /data/python-packages/hrmfast-*.dist-info && pip install --no-cache-dir --no-build-isolation --target=/data/python-packages . && python -c \"
import sys; sys.path.insert(0, '/data/python-packages')
import torch, hrmfast
print('hrmfast import OK')\""

echo "installed into $DATA_ROOT/python-packages"
