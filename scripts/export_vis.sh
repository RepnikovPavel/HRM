#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/mnt/nvme/hrm}"
IMAGE="hrm-visualize:local"

docker build -q -f docker/DockerfileVisualize -t "$IMAGE" docker
docker run --rm --init --user "$(id -u):$(id -g)" \
    -v "$DATA_ROOT:/data" \
    -v "$PWD/dataset/visualize/export_vis.py:/work/export_vis.py:ro" \
    "$IMAGE" python -u /work/export_vis.py --data-root /data/raw-data "$@"
