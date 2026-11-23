#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git gdb \
    python3 python3-venv ca-certificates curl
if [[ "$(llvm-config-20 --version 2>/dev/null || true)" != "20.1.8" ]]; then
sudo install -d -m 0755 /etc/apt/keyrings
curl --fail --location --retry 3 https://apt.llvm.org/llvm-snapshot.gpg.key \
    -o /tmp/miniai-llvm.asc
sudo install -m 0644 /tmp/miniai-llvm.asc /etc/apt/keyrings/llvm.asc
printf '%s\n' 'deb [arch=amd64 signed-by=/etc/apt/keyrings/llvm.asc] https://apt.llvm.org/noble/ llvm-toolchain-noble-20 main' \
    | sudo tee /etc/apt/sources.list.d/miniai-llvm20.list
sudo apt-get update
version='1:20.1.8~++20250804090239+87f0227cb601-1~exp1~20250804210352.139'
sudo apt-get install -y "clang-20=$version" "llvm-20-dev=$version" \
    "libmlir-20-dev=$version" "mlir-20-tools=$version"
fi
if [[ ! -x "$HOME/.venvs/miniai-cpu/bin/python" ]]; then
    python3 -m venv "$HOME/.venvs/miniai-cpu"
fi
"$HOME/.venvs/miniai-cpu/bin/python" -m pip install numpy==2.2.6
llvm-config-20 --version
