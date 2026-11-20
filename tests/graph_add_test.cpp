#include "miniai/Graph.h"

#include <iostream>
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
        Graph graph;
        const NodeId a = graph.input("A", Shape({2, 3}));
        const NodeId b = graph.input("B", Shape({2, 3}));
        const NodeId c = graph.input("C", Shape({2, 3}));

        const NodeId sum = graph.add(a, b);

        require(sum == 3, "Wrong Add node ID");
        require(graph.node(sum).kind == NodeKind::Add,
                "Wrong Add node kind");
        require(graph.node(sum).inputs == std::vector<NodeId>{a, b},
                "Wrong Add dependencies");
        require(graph.node(sum).shape.dimensions() ==
                    std::vector<std::int64_t>{2, 3},
                "Wrong Add result Shape");
        require(graph.node(a).inputs.empty(),
                "Input node unexpectedly has dependencies");

        const NodeId chained = graph.add(sum, c);
        require(graph.node(chained).inputs ==
                    std::vector<NodeId>{sum, c},
                "Wrong chained dependencies");

        const NodeId reused = graph.add(a, a);
        require(graph.node(reused).inputs ==
                    std::vector<NodeId>{a, a},
                "Repeated operand was not preserved");

        const NodeId flat = graph.input("Flat", Shape({6}));
        const NodeId transposed = graph.input("Transposed", Shape({3, 2}));
        const NodeId different = graph.input("Different", Shape({2, 4}));

        const std::size_t before = graph.nodeCount();

        expectThrows<std::invalid_argument>(
            [&] { graph.add(a, flat); },
            "Different ranks were accepted");
        expectThrows<std::invalid_argument>(
            [&] { graph.add(a, transposed); },
            "Different dimensions with equal element counts were accepted");
        expectThrows<std::invalid_argument>(
            [&] { graph.add(a, different); },
            "Different element counts were accepted");
        expectThrows<std::out_of_range>(
            [&] { graph.add(99, a); },
            "Invalid left operand was accepted");
        expectThrows<std::out_of_range>(
            [&] { graph.add(a, 99); },
            "Invalid right operand was accepted");
        expectThrows<std::out_of_range>(
            [&] { graph.add(before, a); },
            "Forward reference was accepted");

        require(graph.nodeCount() == before,
                "Rejected Add operations changed the graph");

        std::cout << "All Add graph tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
