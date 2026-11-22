#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mode="${1:-baseline}"
case "$mode" in
    baseline) options=() ;;
    basic) options=(--opt) ;;
    fused) options=(--fuse) ;;
    *) printf 'Usage: bash run_cpu.sh [baseline|basic|fused]\n' >&2; exit 1 ;;
esac
stage="build/cpu_$mode"
if [[ ! -f build/build.ninja ]]; then
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
fi
cmake --build build --parallel 2 --target miniai-ir miniai_reference
mkdir -p "$stage"
./build/miniai-ir --cpu "${options[@]}" > "$stage/tensor.mlir"
mlir-opt-20 "$stage/tensor.mlir" --inline --symbol-dce -o "$stage/inlined.mlir"
mlir-opt-20 "$stage/inlined.mlir" \
    --empty-tensor-to-alloc-tensor --one-shot-bufferize -o "$stage/bufferized.mlir"
mlir-opt-20 "$stage/bufferized.mlir" \
    --buffer-deallocation-pipeline -o "$stage/deallocated.mlir"
mlir-opt-20 "$stage/deallocated.mlir" \
    --convert-linalg-to-loops -o "$stage/loops.mlir"
mlir-opt-20 "$stage/loops.mlir" \
    --expand-strided-metadata --lower-affine --convert-scf-to-cf \
    --finalize-memref-to-llvm --convert-arith-to-llvm \
    --convert-func-to-llvm --convert-cf-to-llvm \
    --reconcile-unrealized-casts -o "$stage/llvm.mlir"
mlir-translate-20 --mlir-to-llvmir "$stage/llvm.mlir" -o "$stage/kernel.ll"
llc-20 -O0 -filetype=obj -mtriple=x86_64-pc-linux-gnu \
    -relocation-model=pic "$stage/kernel.ll" -o "$stage/kernel.o"
clang++-20 -std=c++17 -O0 -g -Iinclude \
    tools/cpu_demo.cpp "$stage/kernel.o" \
    build/libminiai_reference.a build/libminiai_graph.a -o "$stage/demo"
printf 'CPU mode: %s\n' "$mode"
"./$stage/demo"
