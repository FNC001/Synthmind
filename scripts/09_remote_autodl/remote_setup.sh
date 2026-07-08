#!/usr/bin/env bash
set -euo pipefail

HOST="${AUTODL_HOST:-${1:-}}"
PORT="${AUTODL_PORT:-${2:-}}"
USER_NAME="${AUTODL_USER:-root}"
REMOTE_ROOT="${AUTODL_REMOTE_ROOT:-/root/SynPred_autorun_20260613}"

if [[ -z "${HOST}" || -z "${PORT}" ]]; then
  echo "Usage: AUTODL_HOST=<host> AUTODL_PORT=<port> $0"
  exit 2
fi

SSH_OPTS=(-p "${PORT}" -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o StrictHostKeyChecking=accept-new)

ssh "${SSH_OPTS[@]}" "${USER_NAME}@${HOST}" "REMOTE_ROOT='${REMOTE_ROOT}' bash -s" <<'REMOTE'
set -euo pipefail
mkdir -p "${REMOTE_ROOT}/logs"
REPORT="${REMOTE_ROOT}/REMOTE_SETUP_REPORT.md"
{
  echo "# AutoDL Remote Setup Report"
  echo
  echo "Generated: $(date -Iseconds)"
  echo
  echo "## Host"
  hostname || true
  echo
  echo "## GPU"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "nvidia-smi not found"
  fi
  echo
  echo "## CPU / RAM"
  lscpu 2>/dev/null | sed -n '1,24p' || true
  free -h 2>/dev/null || true
  echo
  echo "## Disk"
  df -h || true
  echo
  echo "## Python / Conda"
  command -v python3 || true
  python3 --version 2>/dev/null || true
  command -v python || true
  python --version 2>/dev/null || true
  command -v conda || true
  conda --version 2>/dev/null || true
  echo
  echo "## CUDA"
  command -v nvcc || true
  nvcc --version 2>/dev/null || true
} > "${REPORT}"
cat "${REPORT}"
REMOTE

