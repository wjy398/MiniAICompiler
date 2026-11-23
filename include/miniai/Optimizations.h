#pragma once
#include "mlir/IR/BuiltinOps.h"
#include <cstddef>
#include <memory>
#include "mlir/Pass/Pass.h"

namespace miniai {
struct OptimizationStats {
    std::size_t genericBefore = 0;
    std::size_t genericAfter = 0;
};
// Upstream canonicalization/CSE plus optional restricted custom fusion.
std::unique_ptr<mlir::Pass> createElementwiseFusionPass();
OptimizationStats optimizeModule(mlir::ModuleOp module, bool enableFusion);
} // namespace miniai
