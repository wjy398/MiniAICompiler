#pragma once

#include "miniai/Graph.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <string>
#include <vector>

namespace miniai {

struct GeneratedModule {
    mlir::OwningOpRef<mlir::ModuleOp> module;
    // Function arguments follow these input names in order.
    std::vector<std::string> inputNames;
};

// The caller's context must outlive the returned module.
// Only nodes reachable from the selected output are emitted.
GeneratedModule generateModule(const Graph &graph, mlir::MLIRContext &context,
                               const std::string &functionName = "main");

// Creates a destination-passing memref wrapper for AOT CPU execution.
// Caller-provided input/output buffers must have separate allocations.
GeneratedModule generateCPUModule(const Graph &graph,
                                  mlir::MLIRContext &context,
                                  const std::string &functionName = "miniai_kernel");

} // namespace miniai
