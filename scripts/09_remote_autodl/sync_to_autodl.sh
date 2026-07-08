#!/usr/bin/env bash
set -euo pipefail

HOST="${AUTODL_HOST:-${1:-}}"
PORT="${AUTODL_PORT:-${2:-}}"
USER_NAME="${AUTODL_USER:-root}"
REMOTE_ROOT="${AUTODL_REMOTE_ROOT:-/root/SynPred_autorun_20260613}"
PROJECT_NAME="${AUTODL_PROJECT_NAME:-SynPred}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -z "${HOST}" || -z "${PORT}" ]]; then
  echo "Usage: AUTODL_HOST=<host> AUTODL_PORT=<port> $0"
  exit 2
fi

ssh -p "${PORT}" -o StrictHostKeyChecking=accept-new "${USER_NAME}@${HOST}" "mkdir -p '${REMOTE_ROOT}/${PROJECT_NAME}' '${REMOTE_ROOT}/logs'"

RSYNC_RSH="ssh -p ${PORT} -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o StrictHostKeyChecking=accept-new"
rsync -azL --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude '.env' \
  --exclude '*.pem' \
  --exclude '*.key' \
  --exclude 'wandb/' \
  --exclude 'tmp/' \
  --exclude 'old_outputs/' \
  --exclude 'Genome/' \
  --exclude 'newdata/' \
  --exclude 'data/raw/' \
  --exclude '*.local_bak_*' \
  --exclude 'data/*.local_bak_*' \
  --exclude 'outputs/*.local_bak_*' \
  --exclude 'runs.local_bak_*' \
  --exclude 'outputs/inference/' \
  --exclude 'data/interim/infer/' \
  --exclude 'outputs/autorun/*/final_package/' \
  --exclude 'outputs/autorun/*/*.tar.gz' \
  -e "${RSYNC_RSH}" \
  "${LOCAL_ROOT}/" "${USER_NAME}@${HOST}:${REMOTE_ROOT}/${PROJECT_NAME}/"

echo "Synced ${LOCAL_ROOT} -> ${USER_NAME}@${HOST}:${REMOTE_ROOT}/${PROJECT_NAME}"
