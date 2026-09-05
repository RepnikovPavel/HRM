#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/mnt/nvme/hrm}"
SERVER_DATA_ROOT="${SERVER_DATA_ROOT:-/mnt/hdd2/hrm}"
SERVER_USER="${SERVER_USER:-user}"

if ping -c1 -W2 -I eno1 192.168.0.1 >/dev/null 2>&1; then
    SSH="ssh -o BindInterface=eno1"
    HOST="192.168.0.1"
else
    SSH="ssh"
    HOST="192.168.1.68"
fi

echo "server: $HOST"
rsync -a -e "$SSH" --relative \
    scripts docker/DockerfileDownload dataset/download_raw.py \
    "$SERVER_USER@$HOST:~/HRM/"
$SSH "$SERVER_USER@$HOST" \
    "cd ~/HRM && DATA_ROOT=$SERVER_DATA_ROOT HF_TOKEN=${HF_TOKEN:-} bash scripts/download_data.sh $*"
DATA_ROOT="$DATA_ROOT" SERVER_DATA_ROOT="$SERVER_DATA_ROOT" \
    SERVER_USER="$SERVER_USER" bash scripts/rsync_from_server.sh
