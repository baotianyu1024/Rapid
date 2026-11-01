#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin=/home/longjiu.bty/miniconda3/envs/het-gmp-rapid/bin/python
for dataset in t180 t200 t510; do
  "$python_bin" "$script_dir/generate.py" "$dataset"
done
"$python_bin" "$script_dir/run_all.py"
"$python_bin" "$script_dir/finish.py"
"$python_bin" "$script_dir/../../code/publish_local_docs.py"
