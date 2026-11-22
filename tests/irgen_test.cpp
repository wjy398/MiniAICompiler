#include "miniai/IRGen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Action>
void expectThrows(Action action, const char *message) {
    try {
        action();
    } catch (const Exception &) {
        return;
    }
    throw std::runtime_error(message);
}

void roundTrip(miniai::GeneratedModule &generated, mlir::MLIRContext &context) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    generated.module->print(stream);
    stream.flush();
    auto parsed = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    require(static_cast<bool>(parsed), "Printed MLIR could not be parsed");
    require(mlir::succeeded(mlir::verify(parsed->getOperation())),
            "Parsed MLIR failed verification");
}

void testExpression(const std::vector<std::int64_t> &dimensions) {
    using namespace miniai;
    Graph graph;
    const auto a = graph.input("A", Shape(dimensions));
    const auto b = graph.input("B", Shape(dimensions));
    const auto c = graph.input("C", Shape(dimensions));
    graph.setOutput(graph.mul(graph.relu(graph.add(a, b)), c));

    mlir::MLIRContext context;
    auto generated = generateModule(graph, context);
    require(generated.inputNames == std::vector<std::string>{"A", "B", "C"},
            "Wrong argument order");
    auto function =
        generated.module->lookupSymbol<mlir::func::FuncOp>("main");
    require(static_cast<bool>(function), "Function was not created");
    require(function.getFunctionType().getNumInputs() == 3,
            "Wrong function argument count");
    require(function.getFunctionType().getNumResults() == 1,
            "Wrong function result count");
    const auto type = mlir::cast<mlir::RankedTensorType>(
        function.getFunctionType().getResult(0));
    require(type.getShape() == llvm::ArrayRef<std::int64_t>(dimensions),
            "Wrong function result Shape");
    require(type.getElementType().isF32(), "Wrong tensor element type");

    unsigned generics = 0, adds = 0, muls = 0, maximums = 0;
    generated.module->walk([&](mlir::linalg::GenericOp operation) {
        ++generics;
        require(operation->getRegion(0).hasOneBlock(),
                "Generic should have one computation block");
        for (auto argument : operation->getRegion(0).front().getArguments()) {
            require(argument.getType().isF32(),
                    "Generic computation argument should be scalar f32");
        }
    });
    generated.module->walk([&](mlir::arith::AddFOp) { ++adds; });
    generated.module->walk([&](mlir::arith::MulFOp) { ++muls; });
    generated.module->walk([&](mlir::arith::MaximumFOp) { ++maximums; });
    require(generics == 3 && adds == 1 && muls == 1 && maximums == 1,
            "Wrong expression operations");
    roundTrip(generated, context);
}

void testSharedResult() {
    using namespace miniai;
    Graph graph;
    const auto a = graph.input("A", Shape({4}));
    const auto b = graph.input("B", Shape({4}));
    const auto c = graph.input("C", Shape({4}));
    const auto sum = graph.add(a, b);
    const auto left = graph.relu(sum);
    const auto right = graph.mul(sum, c);
    graph.setOutput(graph.add(left, right));
    mlir::MLIRContext context;
    auto generated = generateModule(graph, context, "shared");
    unsigned count = 0, firstResultUses = 0;
    generated.module->walk([&](mlir::linalg::GenericOp operation) {
        if (count++ == 0) {
            for (const auto &use : operation.getResult(0).getUses()) {
                (void)use;
                ++firstResultUses;
            }
        }
    });
    require(count == 4, "Shared producer was duplicated");
    require(firstResultUses == 2, "Shared result should have two consumers");
    roundTrip(generated, context);
}

void testUnusedAndPassthrough() {
    using namespace miniai;
    Graph graph;
    const auto a = graph.input("A", Shape({4}));
    const auto unused = graph.input("Unused", Shape({2, 3}));
    (void)graph.mul(unused, unused);
    graph.setOutput(a);
    mlir::MLIRContext context;
    auto generated = generateModule(graph, context, "identity");
    require(generated.inputNames == std::vector<std::string>{"A"},
            "Unreachable input was emitted");
    auto function =
        generated.module->lookupSymbol<mlir::func::FuncOp>("identity");
    require(function.getFunctionType().getNumInputs() == 1,
            "Wrong passthrough argument count");
    auto returned = mlir::cast<mlir::func::ReturnOp>(
        &function.getBody().front().back());
    require(returned->getOperand(0) == function.getArgument(0),
            "Passthrough should return its argument");
    unsigned generics = 0;
    generated.module->walk([&](mlir::linalg::GenericOp) { ++generics; });
    require(generics == 0, "Unreachable computation was emitted");
    roundTrip(generated, context);
}

} // namespace

int main() {
    try {
        testExpression({4});
        testExpression({2, 3});
        testSharedResult();
        testUnusedAndPassthrough();
        miniai::Graph graph;
        mlir::MLIRContext context;
        expectThrows<std::logic_error>(
            [&] { (void)miniai::generateModule(graph, context); },
            "Missing output was accepted");
        const auto a = graph.input("A", miniai::Shape({4}));
        graph.setOutput(a);
        expectThrows<std::invalid_argument>(
            [&] { (void)miniai::generateModule(graph, context, ""); },
            "Empty function name was accepted");
        std::cout << "All IR generation tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
