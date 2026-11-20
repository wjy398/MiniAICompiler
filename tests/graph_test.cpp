#include "miniai/Graph.h"

#include <iostream>
#include <limits>
#include <stdexcept>
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

} // namespace

int main() {
    using namespace miniai;

    try {
        Shape matrix({2, 3});
        require(matrix.rank() == 2, "Wrong rank");
        require(matrix.numElements() == 6, "Wrong element count");
        require(matrix.dimensions() ==
                    std::vector<std::int64_t>{2, 3},
                "Wrong dimensions");
        require(Shape({4}).numElements() == 4, "Wrong vector size");

        Graph graph;
        const NodeId a = graph.input("A", matrix);
        const NodeId b = graph.input("B", matrix);

        require(a == 0 && b == 1, "Wrong node IDs");
        require(graph.nodeCount() == 2, "Wrong node count");
        require(graph.node(a).name == "A", "Wrong input name");
        require(graph.node(b).kind == NodeKind::Input, "Wrong node kind");
        require(graph.node(b).shape.numElements() == 6,
                "Wrong input shape");

        expectThrows<std::invalid_argument>(
            [] { (void)Shape(std::vector<std::int64_t>{}); },
            "Empty Shape was accepted");
        expectThrows<std::invalid_argument>(
            [] { (void)Shape({2, 3, 4}); },
            "Rank 3 was accepted");
        expectThrows<std::invalid_argument>(
            [] { (void)Shape({2, 0}); },
            "Zero dimension was accepted");
        expectThrows<std::invalid_argument>(
            [] { (void)Shape({-1, 3}); },
            "Negative dimension was accepted");
        expectThrows<std::overflow_error>(
            [] {
                (void)Shape({
                    std::numeric_limits<std::int64_t>::max(), 3});
            },
            "Element count overflow was accepted");
        expectThrows<std::invalid_argument>(
            [&] { graph.input("A", matrix); },
            "Duplicate name was accepted");
        expectThrows<std::invalid_argument>(
            [&] { graph.input("", matrix); },
            "Empty name was accepted");
        expectThrows<std::out_of_range>(
            [&] { (void)graph.node(99); },
            "Invalid node ID was accepted");

        require(graph.nodeCount() == 2,
                "Rejected inputs changed the graph");

        std::cout << "All graph tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
