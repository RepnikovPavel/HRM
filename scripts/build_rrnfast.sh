#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/mnt/nvme/hrm}"
IMAGE="${TRAIN_IMAGE:-hrm-rrn-train:cuda12}"

mkdir -p "$DATA_ROOT/python-packages"
docker run --rm --init --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$PWD:/work" -w /work/rrnfast \
    -v "$DATA_ROOT/python-packages:/data/python-packages" \
    "$IMAGE" sh -c "rm -rf /data/python-packages/rrnfast /data/python-packages/rrnfast_backend*.so /data/python-packages/rrnfast-*.dist-info && pip install --no-cache-dir --no-build-isolation --target=/data/python-packages . && python -c \"
import sys; sys.path.insert(0, '/data/python-packages')
import torch, rrnfast
print('rrnfast import OK')\""

echo "installed into $DATA_ROOT/python-packages"
