"""Two experiments: vector addition and fused ReLU(A+B)*C.

The separate ReLU/Mul kernels provide the unfused comparison for experiment 2.
BLOCK is elements per Triton program, not the number of CUDA threads.
"""
import torch
import triton
import triton.language as tl

BLOCK = 256


@triton.jit
def _relu_value(value):
    positive = tl.where(value > 0.0, value, 0.0)
    return tl.where(value != value, value, positive)


@triton.jit
def _add(a, b, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(a + offsets, mask=mask, other=0.0)
    y = tl.load(b + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, x + y, mask=mask)


@triton.jit
def _relu(x, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    value = tl.load(x + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, _relu_value(value), mask=mask)


@triton.jit
def _mul(a, b, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(a + offsets, mask=mask, other=0.0)
    y = tl.load(b + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, x * y, mask=mask)


@triton.jit
def _fused(a, b, c, out, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(a + offsets, mask=mask, other=0.0)
    y = tl.load(b + offsets, mask=mask, other=0.0)
    z = tl.load(c + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, _relu_value(x + y) * z, mask=mask)


def validate(*tensors):
    first = tensors[0]
    for value in tensors:
        if (not value.is_cuda or value.dtype != torch.float32
                or value.ndim != 1 or not value.is_contiguous()
                or value.shape != first.shape or value.device != first.device
                or value.numel() == 0):
            raise ValueError(
                "Expected matching nonempty contiguous 1D float32 CUDA tensors")


# Low-level launches below require previously validated tensors and allocated
# output/scratch buffers. They do no allocation, so benchmark boundaries match.
def launch_add(a, b, out):
    n = a.numel()
    return _add[(triton.cdiv(n, BLOCK),)](
        a, b, out, n, BLOCK=BLOCK, num_warps=4, enable_fp_fusion=False)


def launch_fused(a, b, c, out):
    n = a.numel()
    return _fused[(triton.cdiv(n, BLOCK),)](
        a, b, c, out, n, BLOCK=BLOCK, num_warps=4, enable_fp_fusion=False)


def launch_unfused(a, b, c, tmp1, tmp2, out):
    n = a.numel()
    grid = (triton.cdiv(n, BLOCK),)
    _add[grid](a, b, tmp1, n, BLOCK=BLOCK, num_warps=4,
               enable_fp_fusion=False)
    _relu[grid](tmp1, tmp2, n, BLOCK=BLOCK, num_warps=4,
                enable_fp_fusion=False)
    return _mul[grid](tmp2, c, out, n, BLOCK=BLOCK, num_warps=4,
                      enable_fp_fusion=False)


def vector_add(a, b):
    validate(a, b)
    out = torch.empty_like(a)
    launch_add(a, b, out)
    return out


def fused_expression(a, b, c):
    validate(a, b, c)
    out = torch.empty_like(a)
    launch_fused(a, b, c, out)
    return out
