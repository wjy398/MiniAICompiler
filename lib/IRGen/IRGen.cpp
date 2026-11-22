#include "miniai/IRGen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace miniai {

GeneratedModule generateModule(const Graph &graph, mlir::MLIRContext &context,
                               const std::string &functionName) {
    const NodeId output = graph.output();
    if (functionName.empty()) {
        throw std::invalid_argument("Function name must not be empty");
    }
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::linalg::LinalgDialect>();
    context.loadDialect<mlir::tensor::TensorDialect>();

    std::vector<char> required(graph.nodeCount(), false);
    std::vector<NodeId> pending{output};
    while (!pending.empty()) {
        const NodeId id = pending.back();
        pending.pop_back();
        const Node &node = graph.node(id);
        if (required.at(id)) {
            continue;
        }
        required[id] = true;
        for (NodeId input : node.inputs) {
            pending.push_back(input);
        }
    }

    mlir::OpBuilder builder(&context);
    const auto location = builder.getUnknownLoc();
    const auto f32 = builder.getF32Type();
    auto tensorType = [&](const Shape &shape) {
        return mlir::RankedTensorType::get(shape.dimensions(), f32);
    };

    llvm::SmallVector<mlir::Type> argumentTypes;
    std::vector<NodeId> inputIds;
    std::vector<std::string> inputNames;
    for (NodeId id = 0; id < graph.nodeCount(); ++id) {
        const Node &node = graph.node(id);
        if (required[id] && node.kind == NodeKind::Input) {
            argumentTypes.push_back(tensorType(node.shape));
            inputIds.push_back(id);
            inputNames.push_back(node.name);
        }
    }

    mlir::OwningOpRef<mlir::ModuleOp> module(
        mlir::ModuleOp::create(location));
    builder.setInsertionPointToStart(module->getBody());
    auto functionType = builder.getFunctionType(
        argumentTypes, {tensorType(graph.node(output).shape)});
    auto function = builder.create<mlir::func::FuncOp>(
        location, functionName, functionType);
    mlir::Block *entry = function.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // Frontend NodeId -> MLIR Value. Shared graph results map to one Value.
    std::vector<mlir::Value> values(graph.nodeCount());
    for (std::size_t i = 0; i < inputIds.size(); ++i) {
        values[inputIds[i]] = entry->getArgument(i);
    }

    for (NodeId id = 0; id < graph.nodeCount(); ++id) {
        const Node &node = graph.node(id);
        if (!required[id] || node.kind == NodeKind::Input) {
            continue;
        }
        llvm::SmallVector<mlir::Value> operands;
        for (NodeId input : node.inputs) {
            if (!values.at(input)) {
                throw std::logic_error("Graph dependency has no MLIR Value");
            }
            operands.push_back(values[input]);
        }

        const auto resultType = tensorType(node.shape);
        auto empty = builder.create<mlir::tensor::EmptyOp>(
            location, node.shape.dimensions(), f32);
        const auto identity = mlir::AffineMap::getMultiDimIdentityMap(
            node.shape.rank(), &context);
        llvm::SmallVector<mlir::AffineMap> maps(operands.size() + 1, identity);
        llvm::SmallVector<mlir::utils::IteratorType> iterators(
            node.shape.rank(), mlir::utils::IteratorType::parallel);

        auto operation = builder.create<mlir::linalg::GenericOp>(
            location, mlir::TypeRange{resultType}, mlir::ValueRange(operands),
            mlir::ValueRange{empty.getResult()}, maps, iterators,
            [kind = node.kind](mlir::OpBuilder &nested, mlir::Location loc,
                               mlir::ValueRange arguments) {
                mlir::Value result;
                switch (kind) {
                case NodeKind::Add:
                    result = nested.create<mlir::arith::AddFOp>(
                        loc, arguments[0], arguments[1]).getResult();
                    break;
                case NodeKind::Mul:
                    result = nested.create<mlir::arith::MulFOp>(
                        loc, arguments[0], arguments[1]).getResult();
                    break;
                case NodeKind::ReLU: {
                    auto zero = nested.create<mlir::arith::ConstantFloatOp>(
                        loc, llvm::APFloat(0.0f), nested.getF32Type());
                    result = nested.create<mlir::arith::MaximumFOp>(
                        loc, arguments[0], zero.getResult()).getResult();
                    break;
                }
                default:
                    throw std::logic_error("Unsupported graph operation");
                }
                nested.create<mlir::linalg::YieldOp>(
                    loc, mlir::ValueRange{result});
            });
        values[id] = operation.getResult(0);
    }

    builder.create<mlir::func::ReturnOp>(
        location, mlir::ValueRange{values.at(output)});
    if (mlir::failed(mlir::verify(module->getOperation()))) {
        throw std::runtime_error("Generated MLIR module failed verification");
    }
    return GeneratedModule{std::move(module), std::move(inputNames)};
}

GeneratedModule generateCPUModule(const Graph &graph,
                                  mlir::MLIRContext &context,
                                  const std::string &functionName) {
    if (functionName.empty() || functionName == "miniai_tensor") {
        throw std::invalid_argument("Invalid CPU wrapper function name");
    }
    auto generated = generateModule(graph, context, "miniai_tensor");
    context.loadDialect<mlir::bufferization::BufferizationDialect>();
    context.loadDialect<mlir::memref::MemRefDialect>();
    auto tensorFunction =
        generated.module->lookupSymbol<mlir::func::FuncOp>("miniai_tensor");
    tensorFunction.setPrivate();

    mlir::OpBuilder builder(&context);
    const auto location = builder.getUnknownLoc();
    llvm::SmallVector<mlir::Type> buffers;
    for (mlir::Type type : tensorFunction.getArgumentTypes()) {
        const auto tensor = mlir::cast<mlir::RankedTensorType>(type);
        buffers.push_back(mlir::MemRefType::get(
            tensor.getShape(), tensor.getElementType()));
    }
    const auto outputType = mlir::cast<mlir::RankedTensorType>(
        tensorFunction.getFunctionType().getResult(0));
    buffers.push_back(mlir::MemRefType::get(
        outputType.getShape(), outputType.getElementType()));

    builder.setInsertionPointToEnd(generated.module->getBody());
    auto wrapper = builder.create<mlir::func::FuncOp>(
        location, functionName, builder.getFunctionType(buffers, {}));
    wrapper->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    auto *entry = wrapper.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    llvm::SmallVector<mlir::Value> inputs;
    for (std::size_t i = 0; i < generated.inputNames.size(); ++i) {
        inputs.push_back(builder.create<mlir::bufferization::ToTensorOp>(
            location, entry->getArgument(i),
            /*restrict=*/true, /*writeable=*/false).getResult());
    }
    auto call = builder.create<mlir::func::CallOp>(
        location, tensorFunction, inputs);
    auto store =
        builder.create<mlir::bufferization::MaterializeInDestinationOp>(
            location, call.getResult(0),
            entry->getArgument(generated.inputNames.size()));
    store->setAttr("writable", builder.getUnitAttr());
    builder.create<mlir::func::ReturnOp>(location);

    if (mlir::failed(mlir::verify(generated.module->getOperation()))) {
        throw std::runtime_error("CPU wrapper failed verification");
    }
    return generated;
}

} // namespace miniai
