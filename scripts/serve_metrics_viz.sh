#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/mnt/nvme/hrm}"
IMAGE="${VIZ_IMAGE:-hrm-metrics-viz:local}"
PORT="${PORT:-8377}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -q -f docker/DockerfileMetricsViz -t "$IMAGE" docker >/dev/null
fi

docker rm -f hrm-metrics-viz >/dev/null 2>&1 || true
docker run -d --name hrm-metrics-viz --restart unless-stopped \
    --user "$(id -u):$(id -g)" \
    -e CKPT_DIR=/data/ckpt -e PORT="$PORT" \
    -p "$PORT:$PORT" \
    -v "$PWD:/work:ro" \
    -v "$DATA_ROOT/checkpoints:/data/ckpt:ro" \
    "$IMAGE"
echo "metrics viz: http://localhost:$PORT"
