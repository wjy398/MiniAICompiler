"""AOT compilation and ranked-memref invocation; no PyTorch dependency."""
import ctypes
import functools
import json
from pathlib import Path
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@functools.lru_cache(None)
def descriptor_type(rank):
    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.POINTER(ctypes.c_float)),
            ("aligned", ctypes.POINTER(ctypes.c_float)),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * rank),
            ("strides", ctypes.c_int64 * rank)]
    return Descriptor


class BoundKernel:
    def __init__(self, kernel, supplied):
        self.kernel = kernel
        self.inputs = {}
        for name in kernel.metadata["inputs"]:
            if name not in supplied:
                raise ValueError("Missing input: " + name)
            value = supplied[name]
            if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                    or value.shape != kernel.shape or not value.flags.c_contiguous):
                raise ValueError("Input must match contiguous float32 Shape: " + name)
            # Independent allocations fulfill the restrict contract at to_tensor.
            self.inputs[name] = value.copy(order="C")
        self.output = np.full(kernel.shape, np.nan, dtype=np.float32)
        arrays = [self.inputs[n] for n in kernel.metadata["inputs"]] + [self.output]
        cls = descriptor_type(len(kernel.shape))
        self.descriptors = []
        for array in arrays:
            ptr = array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            self.descriptors.append(cls(
                ptr, ptr, 0, (ctypes.c_int64 * array.ndim)(*array.shape),
                (ctypes.c_int64 * array.ndim)(
                    *(stride // array.itemsize for stride in array.strides))))
        self.arguments = [ctypes.byref(d) for d in self.descriptors]

    def invoke(self):
        self.kernel.function(*self.arguments)
        return self.output


class CompiledKernel:
    def __init__(self, directory, compile_ns):
        self.directory = directory
        self.metadata = json.loads((directory / "metadata.json").read_text())
        self.shape = tuple(self.metadata["shape"])
        self.compile_ns = compile_ns
        self.library = ctypes.CDLL(str(directory / "kernel.so"))
        self.function = getattr(self.library, self.metadata["entry"])
        cls = descriptor_type(len(self.shape))
        self.function.argtypes = [ctypes.POINTER(cls)] * (
            len(self.metadata["inputs"]) + 1)
        self.function.restype = None

    def bind(self, inputs):
        return BoundKernel(self, inputs)


_compiled = {}


def build():
    commands = []
    if not (ROOT / "build/build.ninja").is_file():
        commands.append(["cmake", "-S", ".", "-B", "build", "-G", "Ninja",
                         "-DCMAKE_BUILD_TYPE=Debug"])
    commands.append(["cmake", "--build", "build", "--parallel", "2",
                     "--target", "miniai-ir"])
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


def compile_kernel(shape, graph="expression", mode="fused"):
    shape = tuple(shape)
    if mode not in ("baseline", "basic", "fused"):
        raise ValueError("Invalid mode")
    key = (shape, graph, mode)
    if key in _compiled:
        return _compiled[key]
    directory = ROOT / "build/aot" / (
        graph + "-" + "x".join(map(str, shape)) + "-" + mode)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    log = []

    def run(command, output=None):
        result = subprocess.run(list(map(str, command)), cwd=ROOT,
                                capture_output=True, text=True)
        log.append({"command": list(map(str, command)),
                    "returncode": result.returncode, "stderr": result.stderr})
        (directory / "commands.json").write_text(json.dumps(log, indent=2))
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        if output:
            (directory / output).write_text(result.stdout)

    flag = [] if mode == "baseline" else ["--fuse" if mode == "fused" else "--opt"]
    run([ROOT / "build/miniai-ir", "--cpu", "--shape",
         "x".join(map(str, shape)), "--graph", graph,
         "--metadata", directory / "metadata.json", *flag], "tensor.mlir")
    run(["mlir-opt-20", directory / "tensor.mlir", "--inline", "--symbol-dce",
         "-o", directory / "inlined.mlir"])
    run(["mlir-opt-20", directory / "inlined.mlir",
         "--empty-tensor-to-alloc-tensor", "--one-shot-bufferize",
         "-o", directory / "bufferized.mlir"])
    run(["mlir-opt-20", directory / "bufferized.mlir",
         "--buffer-deallocation-pipeline", "-o", directory / "deallocated.mlir"])
    run(["mlir-opt-20", directory / "deallocated.mlir", "--convert-linalg-to-loops",
         "-o", directory / "loops.mlir"])
    run(["mlir-opt-20", directory / "loops.mlir", "--expand-strided-metadata",
         "--lower-affine", "--convert-scf-to-cf", "--finalize-memref-to-llvm",
         "--convert-arith-to-llvm", "--convert-func-to-llvm",
         "--convert-cf-to-llvm", "--reconcile-unrealized-casts",
         "-o", directory / "llvm.mlir"])
    run(["mlir-translate-20", "--mlir-to-llvmir", directory / "llvm.mlir",
         "-o", directory / "kernel.ll"])
    run(["clang-20", "-O3", "-ffp-contract=off", "-shared", "-fPIC",
         directory / "kernel.ll", "-o", directory / "kernel.so"])
    kernel = CompiledKernel(directory, time.perf_counter_ns() - started)
    _compiled[key] = kernel
    return kernel


def reference(graph, inputs):
    a = inputs["A"]
    if graph == "identity":
        return a.copy()
    if graph == "relu":
        return np.where(np.isnan(a), a, np.where(a > 0, a, np.float32(0)))
    b = inputs["B"]
    if graph == "add":
        return a + b
    if graph == "mul":
        return a * b
    value = a + b
    relu = np.where(np.isnan(value), value,
                    np.where(value > 0, value, np.float32(0)))
    if graph == "shared":
        return relu + value * inputs["C"]
    return relu * inputs["C"]
