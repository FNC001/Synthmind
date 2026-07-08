#!/usr/bin/env bash
set -euo pipefail

HOST="${AUTODL_HOST:-${1:-}}"
PORT="${AUTODL_PORT:-${2:-}}"
USER_NAME="${AUTODL_USER:-root}"
REMOTE_ROOT="${AUTODL_REMOTE_ROOT:-/root/SynPred_autorun_20260613}"
PROJECT_NAME="${AUTODL_PROJECT_NAME:-SynPred}"
REMOTE_OUTPUT="${AUTODL_REMOTE_OUTPUT:-${REMOTE_ROOT}/${PROJECT_NAME}/outputs/autorun/24h_optimization_20260613}"
LOCAL_OUT="${LOCAL_OUT:-outputs/autorun/24h_optimization_20260613_collected}"

if [[ -z "${HOST}" || -z "${PORT}" ]]; then
  echo "Usage: AUTODL_HOST=<host> AUTODL_PORT=<port> $0"
  exit 2
fi

mkdir -p "${LOCAL_OUT}"
RSYNC_RSH="ssh -p ${PORT} -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o StrictHostKeyChecking=accept-new"
rsync -az -e "${RSYNC_RSH}" "${USER_NAME}@${HOST}:${REMOTE_OUTPUT}/" "${LOCAL_OUT}/"
echo "Collected ${REMOTE_OUTPUT} -> ${LOCAL_OUT}"

