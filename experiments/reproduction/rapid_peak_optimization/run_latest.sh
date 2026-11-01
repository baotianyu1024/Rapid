#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${RAPID_PYTHON:-python3}" "$script_dir/run_latest.py" "$@"
