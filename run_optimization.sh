#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
bash run_cpu.sh baseline
bash run_cpu.sh fused
mlir-opt-20 examples/basic_optimizations.mlir \
    --canonicalize --cse -o build/basic.optimized.mlir
python3 - <<'PY'
import json,re
from pathlib import Path
summary={}
for mode in ['baseline','fused']:
    root=Path('build')/('cpu_'+mode)
    tensor=(root/'tensor.mlir').read_text()
    loops=(root/'loops.mlir').read_text()
    buffers=(root/'bufferized.mlir').read_text()
    summary[mode]={
        'linalg_generic':len(re.findall(r'\blinalg\.generic\b',tensor)),
        'scf_for_ops_total':len(re.findall(r'\bscf\.for\b',loops)),
        'memref_alloc_ops_total':len(re.findall(r'\bmemref\.alloc\b',buffers)),
    }
Path('build/optimization_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('IR structure comparison (not a latency benchmark):')
print(json.dumps(summary,indent=2))
print('Basic optimization IR: build/basic.optimized.mlir')
PY
