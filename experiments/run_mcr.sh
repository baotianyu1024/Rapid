#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then echo "Usage: $0 INPUT_BIN FRESH_OUTPUT_DIR (Criteo 2,097,152 rows)"; exit 2; fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${RAPID_PYTHON:-python3}"
"${CXX:-g++}" -O3 -std=c++17 -fopenmp "$here/mcr/mcr_all.cpp" -o "$here/mcr/mcr_all"
"$python_bin" "$here/mcr/mcr_prepare.py" --input "$1" --out "$2"
"$here/mcr/mcr_all" "$1" "$2/frequency.u32" "$2/table_offsets.u64" "$2"
"$python_bin" "$here/mcr/mcr_audit.py" --out "$2"
