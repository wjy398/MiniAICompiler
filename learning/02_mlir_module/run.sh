#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"

cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
    -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir

cmake --build build --parallel 2

./build/mlir_module_demo > build/add.mlir

mlir-opt-20 build/add.mlir \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    -o build/add_llvm.mlir

mlir-translate-20 --mlir-to-llvmir \
    build/add_llvm.mlir -o build/add.ll

llc-20 -O0 -filetype=obj \
    -mtriple=x86_64-pc-linux-gnu \
    build/add.ll -o build/add_kernel.o

clang++-20 -std=c++17 -O0 -g \
    runner.cpp build/add_kernel.o -o build/run_add

./build/run_add
