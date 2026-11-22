#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
bash gpu/install.sh
venv="${MINIAI_GPU_VENV:-$HOME/.venvs/miniai-gpu}"
export TRITON_CACHE_DIR="$PWD/build/triton-cache"
mkdir -p build/gpu "$TRITON_CACHE_DIR"
"$venv/bin/python" gpu/run.py --benchmark --output "$PWD/build/gpu"
