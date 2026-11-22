MiniAICompiler GPU experiments
1. Vector addition; 2. fused ReLU(A+B)*C compared with separate Triton launches.
This is an independent GPU experiment, not automatic GPU lowering of the C++ graph.
BLOCK=256 is elements per program; num_warps=4 is a launch parameter. They are not the same quantity.
Offsets use program_id and arange; load/store masks handle the non-divisible tail.
The scalar helper propagates NaN and maps nonpositive non-NaN values to positive zero.
FP fusion is disabled to preserve the intended add/ReLU/multiply evaluation sequence.
Triton JIT is provided by Triton; the CPU compiler project remains AOT.
CUDA runtime dependencies are supplied by the PyTorch wheel. No Linux CUDA driver is installed.
GPU Python: /home/miniai/.venvs/miniai-gpu. It is reusable and separate from system Python and LLVM/MLIR.
Run: bash run_gpu.sh. Correctness, environment, raw timing samples and TTIR/LLVM IR/PTX go to build/gpu.
Timing: preallocated buffers, 20 warmups, 20 samples of 50 repeated calls, CUDA events.
The measured interval excludes first JIT and allocation but includes possible Python dispatch gaps.
Inputs/buffers are reused; there is no explicit GPU-cache flush or fixed GPU clock.
These are steady-state invocation measurements, not isolated kernel-cycle or DRAM-bandwidth measurements.
Actual observed N=1048589 medians: Torch add 14.408 us, Triton add 17.674 us,
unfused expression 50.460 us, fused expression 18.173 us. No universal speedup is claimed.
References: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
https://pytorch.org/get-started/previous-versions/
