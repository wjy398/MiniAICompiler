import argparse
import csv
import datetime
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
from aot import ROOT, build, compile_kernel, reference


def inputs_for(shape, seed):
    rng = np.random.default_rng(seed)
    return {name: rng.normal(size=shape).astype(np.float32) for name in "ABC"}


def check(kernel, values):
    expected = reference(kernel.metadata["graph"], values)
    bound = kernel.bind(values)
    actual = bound.invoke()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6,
                               equal_nan=True)
    zero = expected == 0
    if np.any(zero):
        assert np.all(actual[zero] == 0)
        np.testing.assert_array_equal(np.signbit(actual[zero]),
                                      np.signbit(expected[zero]))
    for name, value in bound.inputs.items():
        assert value.tobytes() == values[name].tobytes(), "Input was modified"
    return bound


def measure(bound):
    for _ in range(5):
        bound.invoke()
    raw = []
    for _ in range(15):
        start = time.perf_counter_ns()
        for _ in range(20):
            bound.invoke()
        raw.append((time.perf_counter_ns() - start) / 20 / 1000)
    return {"median_us": statistics.median(raw), "samples_us": raw,
            "warmup": 5, "repeats": 20, "samples": 15}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["demo", "validate", "benchmark", "all"])
    args = parser.parse_args()
    build()
    output = ROOT / "build/reports"
    output.mkdir(parents=True, exist_ok=True)
    report = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "python": platform.python_version(), "numpy": np.__version__,
              "machine": platform.machine(),
              "backend": "Clang 20.1.8 -O3 -ffp-contract=off",
              "timing": ("perf_counter_ns per bound C-ABI invocation; input "
                         "copies/allocation and compilation excluded; ctypes "
                         "overhead and kernel temporary allocations included"),
              "validation": [], "benchmarks": []}
    if args.command == "demo":
        values = {"A": np.array([-1, 2, 3, -4, 5, -6], dtype=np.float32).reshape(2, 3),
                  "B": np.array([3, -3, 1, 4, -2, 7], dtype=np.float32).reshape(2, 3),
                  "C": np.array([2, 3, -1, 4, 0, 2], dtype=np.float32).reshape(2, 3)}
        for mode in ["baseline", "fused"]:
            kernel = compile_kernel((2, 3), mode=mode)
            bound = check(kernel, values)
            print(mode, bound.output.tolist(), "matches NumPy")
    if args.command in ("validate", "all"):
        cases = [("expression", (7,)), ("expression", (2, 3)),
                 ("expression", (17, 19)), ("shared", (7,)),
                 ("add", (7,)), ("mul", (7,)), ("relu", (7,)),
                 ("identity", (7,))]
        for index, (graph, shape) in enumerate(cases):
            values = inputs_for(shape, 2026 + index)
            for mode in ["baseline", "fused"]:
                kernel = compile_kernel(shape, graph, mode)
                check(kernel, values)
                report["validation"].append(
                    {"graph": graph, "shape": shape, "mode": mode, "passed": True,
                     "compile_ms": kernel.compile_ns / 1e6})
        special = np.array([-np.inf, -2, -0.0, 0.0, 2, np.inf, np.nan],
                           dtype=np.float32)
        for mode in ["baseline", "fused"]:
            check(compile_kernel((7,), "relu", mode), {"A": special})
        report["relu_special_values"] = "passed"
        binder = compile_kernel((7,), "relu", "fused")
        rejected = []
        for label, bad in [
            ("missing", {}),
            ("dtype", {"A": np.zeros(7, dtype=np.float64)}),
            ("shape", {"A": np.zeros((1, 7), dtype=np.float32)}),
            ("stride", {"A": np.zeros(14, dtype=np.float32)[::2]})]:
            try:
                binder.bind(bad)
            except ValueError:
                rejected.append(label)
            else:
                raise AssertionError("Invalid binding accepted: " + label)
        report["binding_rejections"] = rejected
        print("CPU NumPy comparison: 16 graph/shape/mode cases, special values and binding checks passed")
    if args.command in ("benchmark", "all"):
        for shape in [(65549,), (1024, 1024)]:
            values = inputs_for(shape, 2048)
            for mode in ["baseline", "basic", "fused"]:
                kernel = compile_kernel(shape, mode=mode)
                bound = check(kernel, values)
                row = {"shape": shape, "mode": mode,
                       "compile_ms": kernel.compile_ns / 1e6, **measure(bound)}
                report["benchmarks"].append(row)
                print("CPU", shape, mode, f"{row['median_us']:.3f} us")
    cpuinfo = subprocess.run(["lscpu"], capture_output=True, text=True)
    (output / "cpu_environment.txt").write_text(cpuinfo.stdout)
    report_path = output / ("cpu_" + args.command + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if report["benchmarks"]:
        with (output / "cpu_benchmark.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["shape", "mode", "compile_ms", "median_us"])
            for row in report["benchmarks"]:
                writer.writerow(["x".join(map(str, row["shape"])), row["mode"],
                                 row["compile_ms"], row["median_us"]])
    print("Saved:", report_path)


if __name__ == "__main__":
    main()
