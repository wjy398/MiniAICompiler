#include "miniai/ReferenceExecutor.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

// MLIR's ranked memref descriptor ABI for a rank-2 float32 buffer.
struct MemRef2D {
    float *allocated;
    float *aligned;
    std::int64_t offset;
    std::int64_t sizes[2];
    std::int64_t strides[2];
};

extern "C" void _mlir_ciface_miniai_kernel(
    MemRef2D *a, MemRef2D *b, MemRef2D *c, MemRef2D *output);

MemRef2D descriptor(std::vector<float> &data) {
    return {data.data(), data.data(), 0, {2, 3}, {3, 1}};
}

int main() {
    try {
        std::vector<float> a{-1, 2, 3, -4, 5, -6};
        std::vector<float> b{3, -3, 1, 4, -2, 7};
        std::vector<float> c{2, 3, -1, 4, 0, 2};
        std::vector<float> output(6);
        auto da = descriptor(a);
        auto db = descriptor(b);
        auto dc = descriptor(c);
        auto dy = descriptor(output);

        // Inputs and output have separate allocations. The kernel does not own
        // these buffers; std::vector manages their lifetime.
        miniai::Graph graph;
        const auto x = graph.input("A", miniai::Shape({2, 3}));
        const auto y = graph.input("B", miniai::Shape({2, 3}));
        const auto z = graph.input("C", miniai::Shape({2, 3}));
        graph.setOutput(graph.mul(graph.relu(graph.add(x, y)), z));
        miniai::InputMap inputs{
            {"A", miniai::Tensor(miniai::Shape({2, 3}), a)},
            {"B", miniai::Tensor(miniai::Shape({2, 3}), b)},
            {"C", miniai::Tensor(miniai::Shape({2, 3}), c)}};
        const auto expected = miniai::evaluate(graph, inputs);

        _mlir_ciface_miniai_kernel(&da, &db, &dc, &dy);
        if (a != inputs.at("A").data() || b != inputs.at("B").data() ||
            c != inputs.at("C").data()) {
            throw std::runtime_error("Kernel modified a read-only input");
        }

        std::cout << "Compiled CPU output:";
        for (std::size_t i = 0; i < output.size(); ++i) {
            if (output[i] != expected.data()[i] ||
                (output[i] == 0.0f &&
                 std::signbit(output[i]) != std::signbit(expected.data()[i]))) {
                throw std::runtime_error(
                    "Compiled output differs from reference at element " +
                    std::to_string(i));
            }
            std::cout << ' ' << output[i];
        }
        std::cout << "\nCPU result matches reference\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
