#pragma once

#include "miniai/Graph.h"
#include "miniai/Tensor.h"

#include <string>
#include <unordered_map>

namespace miniai {

using InputMap = std::unordered_map<std::string, Tensor>;

// Evaluates only nodes reachable from the selected output.
// Input tensors are read-only. Missing or mismatched required inputs are errors.
// This is a correctness reference, not the LLVM-compiled execution path.
Tensor evaluate(const Graph &graph, const InputMap &inputs);

} // namespace miniai
