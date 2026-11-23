#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
command="${1:-cpu}"
if [[ "$command" == gpu ]]; then
    bash run_gpu.sh
    exit
fi
if [[ -n "${MINIAI_PYTHON:-}" ]]; then
    python="$MINIAI_PYTHON"
elif [[ -x "$HOME/.venvs/miniai-gpu/bin/python" ]]; then
    python="$HOME/.venvs/miniai-gpu/bin/python"
elif python3 -c 'import numpy' 2>/dev/null; then
    python=python3
else
    python="$HOME/.venvs/miniai-cpu/bin/python"
    if [[ ! -x "$python" ]]; then
        python3 -m venv "$HOME/.venvs/miniai-cpu"
    fi
    if ! "$python" -c 'import numpy' 2>/dev/null; then
        "$python" -m pip install numpy==2.2.6
    fi
fi
case "$command" in
    cpu) "$python" benchmarks/evaluate.py demo ;;
    validate) "$python" benchmarks/evaluate.py validate ;;
    benchmark) "$python" benchmarks/evaluate.py benchmark ;;
    all) "$python" benchmarks/evaluate.py all ;;
    *) printf 'Usage: bash run_project.sh [cpu|gpu|validate|benchmark|all]\n' >&2; exit 1 ;;
esac
