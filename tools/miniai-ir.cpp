#include "miniai/IRGen.h"
#include "miniai/Optimizations.h"
#include "llvm/Support/raw_ostream.h"
#include <exception>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
    try {
        bool cpu = false, optimize = false, fusion = false;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--cpu")
                cpu = true;
            else if (option == "--opt")
                optimize = true;
            else if (option == "--fuse")
                optimize = fusion = true;
            else
                throw std::invalid_argument(
                    "Usage: miniai-ir [--cpu] [--opt|--fuse]");
        }
        miniai::Graph graph;
        const auto a = graph.input("A", miniai::Shape({2, 3}));
        const auto b = graph.input("B", miniai::Shape({2, 3}));
        const auto c = graph.input("C", miniai::Shape({2, 3}));
        graph.setOutput(graph.mul(graph.relu(graph.add(a, b)), c));
        mlir::MLIRContext context;
        auto generated = cpu
            ? miniai::generateCPUModule(graph, context)
            : miniai::generateModule(graph, context);
        if (optimize) {
            const auto stats = miniai::optimizeModule(*generated.module, fusion);
            llvm::errs() << "linalg.generic: " << stats.genericBefore
                         << " -> " << stats.genericAfter
                         << (fusion ? " (fusion enabled)\n"
                                    : " (basic optimizations)\n");
        }
        generated.module->print(llvm::outs());
        llvm::outs() << '\n';
        return 0;
    } catch (const std::exception &error) {
        llvm::errs() << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
