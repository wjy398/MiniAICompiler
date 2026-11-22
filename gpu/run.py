import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

import torch
import triton
from kernels import (
    BLOCK, fused_expression, launch_add, launch_fused, launch_unfused,
    validate, vector_add,
)


def reference(a, b, c):
    value = a + b
    positive = torch.where(value > 0, value, torch.zeros_like(value))
    relu = torch.where(torch.isnan(value), value, positive)
    return relu * c


def measure(function, warmup=20, samples=20, repeats=50):
    # Excludes first compilation and allocation. CUDA-event intervals include
    # steady repeated launches; Python dispatch gaps may contribute.
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            function()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / repeats)
    return {"median_us": statistics.median(values), "samples_us": values,
            "warmup": warmup, "samples": samples, "repeats": repeats}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "build/gpu")
    args = parser.parse_args()
    if os.environ.get("TRITON_INTERPRET") == "1":
        raise RuntimeError("Use a real GPU, not TRITON_INTERPRET")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the CUDA GPU")
    if torch.__version__ != "2.6.0+cu124" or triton.__version__ != "3.2.0":
        raise RuntimeError("Use the project's pinned GPU environment")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    smi = subprocess.run(
        ["/usr/lib/wsl/lib/nvidia-smi",
         "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True)
    environment = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": torch.__version__,
        "triton": triton.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "driver": smi.stdout.strip(), "block": BLOCK, "num_warps": 4,
        "enable_fp_fusion": False,
        "timing": ("CUDA-event interval / repeated calls; allocation and first "
                   "JIT excluded; Python dispatch gaps may contribute"),
    }
    print(f"GPU: {properties.name}")
    print(f"PyTorch {torch.__version__}; Triton {triton.__version__}")

    n = 100003  # deliberately not divisible by BLOCK
    a, b, c = [torch.randn(n, device=device) for _ in range(3)]
    actual_add = vector_add(a, b)
    actual_fused = fused_expression(a, b, c)
    torch.testing.assert_close(actual_add, a + b, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_fused, reference(a, b, c),
                               rtol=1e-5, atol=1e-6, equal_nan=True)
    tmp1, tmp2, unfused = [torch.empty_like(a) for _ in range(3)]
    launch_unfused(a, b, c, tmp1, tmp2, unfused)
    torch.testing.assert_close(unfused, actual_fused, rtol=1e-5, atol=1e-6)
    print(f"Vector add and fused expression: correct, N={n} (tail mask used)")

    special_a = torch.tensor(
        [-float("inf"), -2, -0.0, 0.0, 2, float("inf"), float("nan")],
        dtype=torch.float32, device=device)
    special_b = torch.tensor([0, 0, -0.0, 0, 0, 0, 0],
                             dtype=torch.float32, device=device)
    special = fused_expression(special_a, special_b,
                               torch.ones_like(special_a))
    expected = torch.tensor([0, 0, 0, 0, 2, float("inf"), float("nan")],
                            dtype=torch.float32, device=device)
    torch.testing.assert_close(special, expected, equal_nan=True)
    if torch.signbit(special[:4]).any().item():
        raise RuntimeError("ReLU should produce positive zero")
    print("ReLU NaN/infinity/signed-zero semantics: correct")

    artifact = launch_fused(a, b, c, actual_fused)
    torch.cuda.synchronize()
    artifact_dir = args.output / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    artifact_names = []
    for key, suffix in [("ttir", "ttir.mlir"), ("ttgir", "ttgir.mlir"),
                        ("llir", "ll"), ("ptx", "ptx")]:
        if key in artifact.asm and isinstance(artifact.asm[key], str):
            path = artifact_dir / ("fused." + suffix)
            path.write_text(artifact.asm[key])
            artifact_names.append(path.name)
    if "fused.ptx" not in artifact_names:
        raise RuntimeError("GPU PTX artifact was not produced")

    rows = []
    if args.benchmark:
        for size in [65549, 1048589]:
            a, b, c = [torch.randn(size, device=device) for _ in range(3)]
            out_add, out_fused, out_unfused, out_torch, tmp1, tmp2 = [
                torch.empty_like(a) for _ in range(6)]
            validate(a, b, c, out_add, out_fused, out_unfused, out_torch,
                     tmp1, tmp2)
            functions = [
                ("vector_add", "pytorch", lambda: torch.add(a, b, out=out_torch)),
                ("vector_add", "triton", lambda: launch_add(a, b, out_add)),
                ("expression", "triton_unfused",
                 lambda: launch_unfused(a, b, c, tmp1, tmp2, out_unfused)),
                ("expression", "triton_fused",
                 lambda: launch_fused(a, b, c, out_fused)),
            ]
            for experiment, implementation, function in functions:
                function()
                actual = {"pytorch": out_torch, "triton": out_add,
                          "triton_unfused": out_unfused,
                          "triton_fused": out_fused}[implementation]
                expected = a + b if experiment == "vector_add" else reference(a, b, c)
                torch.testing.assert_close(actual, expected,
                                           rtol=1e-5, atol=1e-6, equal_nan=True)
                measurement = measure(function)
                row = {"experiment": experiment, "implementation": implementation,
                       "n": size, **measurement}
                rows.append(row)
                print(f"{experiment:10s} {implementation:15s} N={size:7d} "
                      f"median={row['median_us']:.3f} us")

    result = {"environment": environment, "validation": {
        "n": n, "vector_add": "passed", "fused_expression": "passed",
        "unfused_agreement": "passed", "relu_special_values": "passed"},
        "artifacts": artifact_names, "benchmarks": rows}
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n")
    if rows:
        with (args.output / "benchmark.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["experiment", "implementation", "n", "median_us"])
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in writer.fieldnames})
    print(f"GPU experiments complete; results and compiler artifacts: {args.output}")


if __name__ == "__main__":
    main()
