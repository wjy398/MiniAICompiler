#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

int main() {
    mlir::MLIRContext context;

    mlir::OwningOpRef<mlir::ModuleOp> module(
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));

    if (mlir::failed(mlir::verify(module->getOperation()))) {
        llvm::errs() << "Module verification failed\n";
        return 1;
    }

    module->print(llvm::outs());
    llvm::outs() << '\n';

    return 0;
}
