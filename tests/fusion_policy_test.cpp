#include "miniai/IRGen.h"
#include "miniai/Optimizations.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include <iostream>
#include <stdexcept>

unsigned count(mlir::ModuleOp module) {
    unsigned result = 0;
    module.walk([&](mlir::linalg::GenericOp) { ++result; });
    return result;
}
void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
void run(mlir::ModuleOp module) {
    mlir::PassManager manager(module.getContext());
    manager.addNestedPass<mlir::func::FuncOp>(
        miniai::createElementwiseFusionPass());
    require(mlir::succeeded(manager.run(module)), "Fusion pass failed");
    require(mlir::succeeded(mlir::verify(module)), "Invalid transformed IR");
}

int main() {
    try {
        mlir::MLIRContext context;
        miniai::Graph chain;
        const auto a = chain.input("A", miniai::Shape({4}));
        const auto b = chain.input("B", miniai::Shape({4}));
        const auto c = chain.input("C", miniai::Shape({4}));
        chain.setOutput(chain.mul(chain.relu(chain.add(a, b)), c));
        auto first = miniai::generateModule(chain, context);
        run(*first.module);
        require(count(*first.module) == 1, "Eligible chain did not fuse");

        miniai::Graph shared;
        const auto x = shared.input("A", miniai::Shape({4}));
        const auto y = shared.input("B", miniai::Shape({4}));
        const auto z = shared.input("C", miniai::Shape({4}));
        const auto sum = shared.add(x, y);
        shared.setOutput(shared.add(shared.relu(sum), shared.mul(sum, z)));
        auto second = miniai::generateModule(shared, context);
        run(*second.module);
        require(count(*second.module) == 2,
                "Shared producer should remain separate");

        const char *rank3 = R"mlir(
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @rank3(%a: tensor<2x2x2xf32>, %b: tensor<2x2x2xf32>)
      -> tensor<2x2x2xf32> {
    %empty = tensor.empty() : tensor<2x2x2xf32>
    %t = linalg.generic {indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%a, %b : tensor<2x2x2xf32>, tensor<2x2x2xf32>)
      outs(%empty : tensor<2x2x2xf32>) {
      ^bb0(%x: f32, %y: f32, %out: f32):
        %v = arith.addf %x, %y : f32
        linalg.yield %v : f32
    } -> tensor<2x2x2xf32>
    %r = linalg.generic {indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%t, %b : tensor<2x2x2xf32>, tensor<2x2x2xf32>)
      outs(%empty : tensor<2x2x2xf32>) {
      ^bb0(%x: f32, %y: f32, %out: f32):
        %v = arith.mulf %x, %y : f32
        linalg.yield %v : f32
    } -> tensor<2x2x2xf32>
    return %r : tensor<2x2x2xf32>
  }
})mlir";
        auto third = mlir::parseSourceString<mlir::ModuleOp>(rank3, &context);
        require(static_cast<bool>(third), "Invalid rank-3 fixture");
        run(*third);
        require(count(*third) == 2, "Unsupported rank was fused");
        std::cout << "Fusion eligibility and rejection cases passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
