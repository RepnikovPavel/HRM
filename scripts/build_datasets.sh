#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/mnt/nvme/hrm}"
IMAGE="hrm-build:local"

mkdir -p "$DATA_ROOT/built"
docker build -q -f docker/DockerfileBuild -t "$IMAGE" docker

run_build() {
    echo "=== build $1 -> $2"
    shift 2
    docker run --rm --init --user "$(id -u):$(id -g)" \
        -e OMP_NUM_THREADS="$(nproc)" \
        -v "$DATA_ROOT:/data" \
        -v "$PWD/dataset:/work:ro" \
        -w /work \
        "$IMAGE" python -u "$@"
}

run_build sudoku-extreme-full "$DATA_ROOT/built" \
    build_sudoku_dataset.py --source-dir /data/raw-data/sudoku-extreme \
    --output-dir /data/built/sudoku-extreme-full

run_build sudoku-extreme-1k-aug-1000 "$DATA_ROOT/built" \
    build_sudoku_dataset.py --source-dir /data/raw-data/sudoku-extreme \
    --output-dir /data/built/sudoku-extreme-1k-aug-1000 \
    --subsample-size 1000 --num-aug 1000

run_build maze-30x30-hard-1k "$DATA_ROOT/built" \
    build_maze_dataset.py --source-dir /data/raw-data/maze-30x30-hard-1k \
    --output-dir /data/built/maze-30x30-hard-1k

echo "=== done"
du -sm "$DATA_ROOT"/built/*
