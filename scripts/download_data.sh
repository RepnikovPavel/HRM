#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/mnt/nvme/hrm}"
IMAGE="hrm-download:local"

mkdir -p "$DATA_ROOT/raw-data"
docker build -q -f docker/DockerfileDownload -t "$IMAGE" docker
docker run --rm --init --user "$(id -u):$(id -g)" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "$DATA_ROOT:/data" \
    -v "$PWD/dataset/download_raw.py:/work/download_raw.py:ro" \
    "$IMAGE" python -u /work/download_raw.py --output-dir /data/raw-data "$@"
