#!/usr/bin/env bash
set -euo pipefail

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
mkdir -p "$DATA_ROOT/raw-data"
rsync -a --info=progress2 -e "$SSH" \
    "$SERVER_USER@$HOST:$SERVER_DATA_ROOT/raw-data/" "$DATA_ROOT/raw-data/"
