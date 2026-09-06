#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/mnt/nvme/hrm}"
IMAGE="${TRAIN_IMAGE:-hrm-train:cuda12}"
NGPU="${NGPU:-1}"
DATASET="${1:?dataset path, e.g. /data/built/sudoku-extreme-1k-aug-1000}"
GBS="${2:?global batch size}"
SECS="${3:-150}"
RUN="bench-$(basename "$DATASET")-$GBS"

mkdir -p "$DATA_ROOT/checkpoints/$RUN" "$DATA_ROOT/torch-cache" "$DATA_ROOT/triton-cache"
rm -f "$DATA_ROOT/checkpoints/$RUN/metrics.jsonl"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -q -f docker/DockerfileTrain -t "$IMAGE" docker >/dev/null
fi

CID=$(docker run -d --gpus all --ipc=host --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e DISABLE_COMPILE -e COMPILE_MODE -e TORCHINDUCTOR_COORDINATE_DESCENT_TUNING -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e METRICS_FLUSH -e PYTHONPATH=/data/python-packages -e OMP_NUM_THREADS=8 \
    -e TORCHINDUCTOR_CACHE_DIR=/data/torch-cache -e TRITON_CACHE_DIR=/data/triton-cache \
    -v "$PWD:/work" -w /work \
    -v "$DATA_ROOT/built:/data/built:ro" \
    -v "$DATA_ROOT/checkpoints:/data/checkpoints" \
    -v "$DATA_ROOT/torch-cache:/data/torch-cache" \
    -v "$DATA_ROOT/python-packages:/data/python-packages:ro" \
    -v "$DATA_ROOT/triton-cache:/data/triton-cache" \
    "$IMAGE" torchrun --nproc-per-node "$NGPU" pretrain.py \
    checkpoint_path="/data/checkpoints/$RUN" data_path="$DATASET" \
    epochs=1000000 eval_interval=1000000 global_batch_size="$GBS" lr=1e-4 puzzle_emb_lr=1e-4)

peak=0
peak_w=0
for _ in $(seq 1 $((SECS / 5))); do
    sleep 5
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$m" -gt "$peak" ] && peak=$m
    w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | sort -n | tail -1)
    peak_w=$(echo "$w $peak_w" | awk '{print ($1 > $2) ? $1 : $2}')
done
docker rm -f "$CID" >/dev/null

f="$DATA_ROOT/checkpoints/$RUN/metrics.jsonl"
if [ -f "$f" ]; then
    python3 - "$f" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
if len(rows) > 1:
    sps = (rows[-1]["step"] - rows[0]["step"]) / (rows[-1]["time"] - rows[0]["time"])
    print(f"steps {rows[0]['step']}..{rows[-1]['step']}, {sps:.2f} steps/s")
else:
    print("not enough steps logged")
EOF
fi
echo "PEAK_MEM_MIB=$peak"
echo "PEAK_POWER_W=$peak_w"
