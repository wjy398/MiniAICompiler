#!/usr/bin/env bash
set -euo pipefail
venv="${MINIAI_GPU_VENV:-$HOME/.venvs/miniai-gpu}"
if [[ ! -x "$venv/bin/python" ]]; then
    python3 -m venv "$venv"
fi
if ! "$venv/bin/python" -c 'import torch,triton,numpy; assert torch.__version__=="2.6.0+cu124" and triton.__version__=="3.2.0" and numpy.__version__=="2.2.6"' 2>/dev/null; then
    script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
    "$venv/bin/python" -m pip install --no-cache-dir \
        --index-url https://pypi.org/simple \
        -r "$script_dir/requirements.lock.txt"
fi
printf 'GPU Python environment: %s\n' "$venv"
