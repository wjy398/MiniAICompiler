#include "miniai/Optimizations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <stdexcept>
#include <utility>

namespace miniai {
namespace {

// Our policy: static dense rank-1/2 f32 tensors, identity maps, all-parallel
// iteration, no output accumulation, and only permitted pure scalar operations.
bool isSupported(mlir::linalg::GenericOp op) {
    if (!op.hasPureTensorSemantics() || op->getNumResults() != 1 ||
        op.getNumDpsInits() != 1 ||
        op.getNumParallelLoops() != op.getNumLoops())
        return false;
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(op.getResult(0).getType());
    if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
        type.getRank() < 1 || type.getRank() > 2 || type.getEncoding())
        return false;
    for (mlir::Value operand : op->getOperands())
        if (operand.getType() != type)
            return false;
    for (mlir::AffineMap map : op.getIndexingMapsArray())
        if (!map.isIdentity())
            return false;
    auto &region = op->getRegion(0);
    if (!region.hasOneBlock())
        return false;
    auto &body = region.front();
    if (body.getNumArguments() != op->getNumOperands() ||
        !body.getArguments().back().use_empty())
        return false;
    for (mlir::Operation &nested : body) {
        if (!mlir::isa<mlir::arith::AddFOp, mlir::arith::MulFOp,
                       mlir::arith::MaximumFOp, mlir::arith::ConstantOp,
                       mlir::linalg::YieldOp>(&nested))
            return false;
        for (mlir::Type t : nested.getOperandTypes())
            if (!t.isF32())
                return false;
        for (mlir::Type t : nested.getResultTypes())
            if (!t.isF32())
                return false;
    }
    return true;
}

class FuseSingleUseProducer
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
public:
    using OpRewritePattern::OpRewritePattern;
    mlir::LogicalResult matchAndRewrite(
        mlir::linalg::GenericOp consumer,
        mlir::PatternRewriter &rewriter) const override {
        if (!isSupported(consumer))
            return mlir::failure();
        for (mlir::OpOperand *operand : consumer.getDpsInputOperands()) {
            auto producer =
                operand->get().getDefiningOp<mlir::linalg::GenericOp>();
            if (!producer || !isSupported(producer) ||
                producer.getResult(0).getType() != consumer.getResult(0).getType() ||
                !producer.getResult(0).hasOneUse())
                continue;
            if (!mlir::linalg::areElementwiseOpsFusable(operand))
                continue;
            // Reuse upstream region cloning and SSA remapping, after our
            // conservative checks. Shared/multiple-use producers are rejected.
            auto fused = mlir::linalg::fuseElementwiseOps(rewriter, operand);
            if (mlir::failed(fused))
                return rewriter.notifyMatchFailure(consumer, "fusion failed");
            llvm::SmallVector<mlir::Value> replacements;
            for (mlir::Value value : consumer->getResults())
                replacements.push_back(fused->replacements.lookup(value));
            rewriter.replaceOp(consumer, replacements);
            if (producer->use_empty())
                rewriter.eraseOp(producer);
            return mlir::success();
        }
        return mlir::failure();
    }
};

class ElementwiseFusionPass
    : public mlir::PassWrapper<ElementwiseFusionPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ElementwiseFusionPass)
    llvm::StringRef getArgument() const final {
        return "miniai-fuse-elementwise";
    }
    llvm::StringRef getDescription() const final {
        return "Fuse restricted static float32 elementwise tensor chains";
    }
    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<FuseSingleUseProducer>(&getContext());
        if (mlir::failed(mlir::applyPatternsGreedily(
                getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

std::size_t countGenerics(mlir::ModuleOp module) {
    std::size_t count = 0;
    module.walk([&](mlir::linalg::GenericOp) { ++count; });
    return count;
}
} // namespace

std::unique_ptr<mlir::Pass> createElementwiseFusionPass() {
    return std::make_unique<ElementwiseFusionPass>();
}

OptimizationStats optimizeModule(mlir::ModuleOp module, bool enableFusion) {
    OptimizationStats stats;
    stats.genericBefore = countGenerics(module);
    mlir::PassManager manager(module.getContext());
    manager.enableVerifier(true);
    manager.addPass(mlir::createCanonicalizerPass());
    manager.addPass(mlir::createCSEPass());
    if (enableFusion)
        manager.addNestedPass<mlir::func::FuncOp>(
            createElementwiseFusionPass());
    manager.addPass(mlir::createCanonicalizerPass());
    manager.addPass(mlir::createCSEPass());
    if (mlir::failed(manager.run(module)))
        throw std::runtime_error("Optimization pipeline failed");
    stats.genericAfter = countGenerics(module);
    return stats;
}
} // namespace miniai
