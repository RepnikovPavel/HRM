#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/mnt/nvme/hrm}"
IMAGE="hrm-report:local"

mkdir -p "$DATA_ROOT/checkpoints-server"
rsync -a -e "ssh -o BindInterface=eno1" \
    user@192.168.0.1:/mnt/hdd2/hrm/checkpoints/ "$DATA_ROOT/checkpoints-server/" || true

docker build -q -f docker/DockerfileReport -t "$IMAGE" docker
docker run --rm --init --user "$(id -u):$(id -g)" \
    -v "$DATA_ROOT/checkpoints:/data/checkpoints-dev:ro" \
    -v "$DATA_ROOT/checkpoints-server:/data/checkpoints-server:ro" \
    -v "$PWD/scripts:/work/scripts:ro" \
    -v "$PWD/reports:/work/reports:rw" \
    -w /work \
    "$IMAGE" python -u scripts/make_report.py \
    /data/checkpoints-dev /data/checkpoints-server /work/reports/hrm_reproduction.html
