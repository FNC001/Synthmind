#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-}"
if test -z "$ROOT" || ! test -d "$ROOT/00_OVERVIEW_AND_MANIFEST"; then
  echo "Usage: $0 /path/to/synthmind_release" >&2
  exit 2
fi

find "$ROOT" -type d -exec chmod 0755 {} +
find "$ROOT" -type f -exec chmod 0644 {} +
find "$ROOT" -type f -name '*.sh' -exec chmod 0755 {} +

if test "$(id -u)" = 0; then
  chown -R root:root "$ROOT"
fi

echo "normalized directories=0755 regular_files=0644 shell_entries=0755 root=$ROOT"
