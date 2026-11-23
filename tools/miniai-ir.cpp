#include "miniai/IRGen.h"
#include "miniai/Optimizations.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    try {
        bool cpu = false, optimize = false, fusion = false;
        std::string kind = "expression", metadataPath;
        std::vector<std::int64_t> dimensions{2, 3};
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--cpu") cpu = true;
            else if (option == "--opt") optimize = true;
            else if (option == "--fuse") optimize = fusion = true;
            else if ((option == "--shape" || option == "--graph" ||
                      option == "--metadata") && i + 1 < argc) {
                const std::string value = argv[++i];
                if (option == "--graph") kind = value;
                else if (option == "--metadata") metadataPath = value;
                else {
                    dimensions.clear();
                    std::size_t start = 0;
                    do {
                        const auto end = value.find('x', start);
                        const auto part = value.substr(start, end - start);
                        std::size_t consumed = 0;
                        const auto dimension = std::stoll(part, &consumed);
                        if (consumed != part.size())
                            throw std::invalid_argument("Invalid Shape");
                        dimensions.push_back(dimension);
                        if (end == std::string::npos) break;
                        start = end + 1;
                    } while (true);
                }
            } else if (option == "--help") {
                llvm::outs() << "miniai-ir [--cpu] [--opt|--fuse] "
                    "[--shape 1024|2x3] [--metadata file.json] "
                    "[--graph expression|shared|add|mul|relu|identity]\n";
                return 0;
            } else throw std::invalid_argument("Invalid option; use --help");
        }
        miniai::Shape shape(dimensions);
        miniai::Graph graph;
        const auto a = graph.input("A", shape);
        const auto b = graph.input("B", shape);
        const auto c = graph.input("C", shape);
        miniai::NodeId output;
        if (kind == "expression")
            output = graph.mul(graph.relu(graph.add(a, b)), c);
        else if (kind == "shared") {
            const auto sum = graph.add(a, b);
            output = graph.add(graph.relu(sum), graph.mul(sum, c));
        } else if (kind == "add") output = graph.add(a, b);
        else if (kind == "mul") output = graph.mul(a, b);
        else if (kind == "relu") output = graph.relu(a);
        else if (kind == "identity") output = a;
        else throw std::invalid_argument("Unknown built-in graph");
        graph.setOutput(output);
        mlir::MLIRContext context;
        auto generated = cpu ? miniai::generateCPUModule(graph, context)
                             : miniai::generateModule(graph, context);
        if (optimize) {
            const auto stats = miniai::optimizeModule(*generated.module, fusion);
            llvm::errs() << "linalg.generic: " << stats.genericBefore
                         << " -> " << stats.genericAfter << '\n';
        }
        if (!metadataPath.empty()) {
            llvm::json::Array names, dims;
            for (const auto &name : generated.inputNames) names.push_back(name);
            for (auto dimension : dimensions) dims.push_back(dimension);
            llvm::json::Object metadata{
                {"inputs", std::move(names)}, {"shape", std::move(dims)},
                {"entry", "_mlir_ciface_miniai_kernel"}, {"graph", kind}};
            std::error_code error;
            llvm::raw_fd_ostream file(metadataPath, error);
            if (error) throw std::runtime_error(error.message());
            llvm::json::OStream json(file, 2);
            json.value(std::move(metadata));
        }
        generated.module->print(llvm::outs());
        llvm::outs() << '\n';
        return 0;
    } catch (const std::exception &error) {
        llvm::errs() << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
