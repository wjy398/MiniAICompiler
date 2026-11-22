#pragma once
#include "mlir/IR/BuiltinOps.h"
#include <cstddef>

namespace miniai {
struct OptimizationStats {
    std::size_t genericBefore = 0;
    std::size_t genericAfter = 0;
};
// Upstream canonicalization/CSE plus optional restricted custom fusion.
OptimizationStats optimizeModule(mlir::ModuleOp module, bool enableFusion);
} // namespace miniai
