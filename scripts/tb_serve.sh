#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/mnt/nvme/hrm}"
RUN_DIR="${1:?usage: tb_serve.sh <checkpoint-dir> [port]}"
PORT="${2:-6006}"
IMAGE="hrm-tb:local"

mkdir -p "$DATA_ROOT/tb"
docker build -q -f docker/DockerfileTensorboard -t "$IMAGE" docker
docker rm -f hrm-tb >/dev/null 2>&1 || true
docker run -d --init --name hrm-tb --user "$(id -u):$(id -g)" \
    -p "$PORT:$PORT" \
    -v "$DATA_ROOT:/data" \
    -v "$PWD/scripts/jsonl_to_tb.py:/work/jsonl_to_tb.py:ro" \
    "$IMAGE" sh -c "python -u /work/jsonl_to_tb.py '/data/$RUN_DIR/metrics.jsonl' /data/tb & sleep 3; exec tensorboard --logdir /data/tb --host 0.0.0.0 --port $PORT"
docker update --restart unless-stopped hrm-tb >/dev/null
echo "tensorboard: http://<host>:$PORT  (container hrm-tb)"
