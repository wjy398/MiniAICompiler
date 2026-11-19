#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

int main() {
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();

    mlir::OpBuilder builder(&context);
    auto location = builder.getUnknownLoc();

    mlir::OwningOpRef<mlir::ModuleOp> module(
        mlir::ModuleOp::create(location));

    builder.setInsertionPointToStart(module->getBody());

    auto f32 = builder.getF32Type();
    auto functionType = builder.getFunctionType({f32, f32}, {f32});
    auto function = builder.create<mlir::func::FuncOp>(
        location, "add", functionType);

    mlir::Block *entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    mlir::Value lhs = entryBlock->getArgument(0);
    mlir::Value rhs = entryBlock->getArgument(1);

    mlir::Value sum =
        builder.create<mlir::arith::AddFOp>(location, lhs, rhs).getResult();

    builder.create<mlir::func::ReturnOp>(
        location, mlir::ValueRange{sum});

    if (mlir::failed(mlir::verify(module->getOperation()))) {
        llvm::errs() << "Module verification failed\n";
        return 1;
    }

    module->print(llvm::outs());
    llvm::outs() << '\n';

    return 0;
}
