#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${RAPID_PYTHON:-python3}" "$root/scripts/run_offline.py" "$@"
