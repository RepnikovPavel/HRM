#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/mnt/nvme/hrm}"
IMAGE="hrm-visualize:local"
PORT="${PORT:-8971}"

while ss -tln | grep -q ":$PORT "; do
    PORT=$((PORT + 1))
done

docker build -q -f docker/DockerfileVisualize -t "$IMAGE" docker
echo "viewer: http://localhost:$PORT/viewer/  (Ctrl-C to stop)"
docker run --rm --init --user "$(id -u):$(id -g)" \
    -p "$PORT:$PORT" \
    -v "$DATA_ROOT/raw-data:/data:ro" \
    -v "$PWD/dataset/visualize/viewer:/viewer:ro" \
    -v "$PWD/dataset/visualize/serve.py:/work/serve.py:ro" \
    "$IMAGE" python -u /work/serve.py --port "$PORT"
