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
        require(!graph.hasOutput(), "Empty graph unexpectedly has output");
        expectThrows<std::logic_error>(
            [&] { (void)graph.output(); },
            "Missing output was accepted");
        expectThrows<std::out_of_range>(
            [&] { graph.setOutput(0); },
            "Output of empty graph was accepted");
        require(!graph.hasOutput(), "Rejected output changed empty graph");

        const NodeId a = graph.input("A", Shape({2, 3}));
        const NodeId b = graph.input("B", Shape({2, 3}));
        const NodeId c = graph.input("C", Shape({2, 3}));
        const NodeId sum = graph.add(a, b);
        const NodeId activated = graph.relu(sum);
        const NodeId y = graph.mul(activated, c);

        require(graph.nodeCount() == 6, "Wrong pipeline node count");
        require(!graph.hasOutput(), "Node creation implicitly selected output");
        expectThrows<std::logic_error>(
            [&] { (void)graph.output(); },
            "Nonempty graph with missing output was accepted");

        require(graph.node(activated).kind == NodeKind::ReLU,
                "Wrong ReLU node kind");
        require(graph.node(activated).inputs == std::vector<NodeId>{sum},
                "Wrong ReLU dependency");
        require(graph.node(activated).shape.dimensions() ==
                    std::vector<std::int64_t>{2, 3},
                "Wrong ReLU result Shape");
        require(graph.node(y).kind == NodeKind::Mul, "Wrong Mul node kind");
        require(graph.node(y).inputs == std::vector<NodeId>{activated, c},
                "Wrong Mul dependencies");
        require(graph.node(y).shape.dimensions() ==
                    std::vector<std::int64_t>{2, 3},
                "Wrong Mul result Shape");

        graph.setOutput(y);
        require(graph.hasOutput() && graph.output() == y,
                "Wrong selected output");
        graph.setOutput(a);
        require(graph.output() == a, "Input passthrough output was rejected");
        graph.setOutput(y);

        const NodeId repeated = graph.mul(a, a);
        require(graph.node(repeated).inputs == std::vector<NodeId>{a, a},
                "Repeated Mul operand was not preserved");
        require(graph.output() == y, "Appending node changed output");

        const NodeId flat = graph.input("Flat", Shape({6}));
        const NodeId reordered = graph.input("Reordered", Shape({3, 2}));
        const NodeId different = graph.input("Different", Shape({2, 4}));
        const std::size_t before = graph.nodeCount();

        expectThrows<std::invalid_argument>(
            [&] { graph.mul(a, flat); }, "Mul accepted different ranks");
        expectThrows<std::invalid_argument>(
            [&] { graph.mul(a, reordered); },
            "Mul accepted different Shapes with equal element counts");
        expectThrows<std::invalid_argument>(
            [&] { graph.mul(a, different); },
            "Mul accepted different element counts");
        expectThrows<std::out_of_range>(
            [&] { graph.mul(99, a); }, "Mul accepted invalid left operand");
        expectThrows<std::out_of_range>(
            [&] { graph.mul(a, 99); }, "Mul accepted invalid right operand");
        expectThrows<std::out_of_range>(
            [&] { graph.relu(99); }, "ReLU accepted invalid operand");
        expectThrows<std::out_of_range>(
            [&] { graph.relu(before); }, "ReLU accepted forward reference");
        expectThrows<std::out_of_range>(
            [&] { graph.setOutput(before); }, "Invalid output was accepted");
        require(graph.nodeCount() == before, "Rejected operations changed graph");
        require(graph.output() == y, "Rejected output replaced valid output");

        Graph vectorGraph;
        const NodeId x = vectorGraph.input("X", Shape({4}));
        const NodeId r = vectorGraph.relu(x);
        const NodeId m = vectorGraph.mul(x, r);
        vectorGraph.setOutput(m);
        require(vectorGraph.node(m).shape.dimensions() ==
                    std::vector<std::int64_t>{4},
                "Wrong one-dimensional result Shape");
        require(vectorGraph.output() == m, "Wrong vector graph output");

        std::cout << "All elementwise graph tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
