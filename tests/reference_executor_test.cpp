#include "miniai/ReferenceExecutor.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string &message) {
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

void check(const miniai::Tensor &actual, const miniai::Shape &shape,
           const std::vector<float> &expected, const char *name) {
    require(actual.shape().dimensions() == shape.dimensions(),
            std::string(name) + ": wrong result Shape");
    require(actual.data().size() == expected.size(),
            std::string(name) + ": wrong result length");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float value = actual.data()[i];
        const std::string where =
            std::string(name) + ": element " + std::to_string(i);
        if (std::isnan(expected[i])) {
            require(std::isnan(value), where + " should be NaN");
        } else {
            require(value == expected[i], where + " has wrong value");
            if (expected[i] == 0.0f) {
                require(std::signbit(value) == std::signbit(expected[i]),
                        where + " has wrong zero sign");
            }
        }
    }
}

void testAddMulAndPassthrough() {
    using namespace miniai;
    Graph graph;
    const auto a = graph.input("A", Shape({4}));
    const auto b = graph.input("B", Shape({4}));
    InputMap inputs{
        {"A", Tensor(Shape({4}), {-2, 0, 1.5f, 3})},
        {"B", Tensor(Shape({4}), {1, 2, 2.25f, -5})}};
    graph.setOutput(graph.add(a, b));
    check(evaluate(graph, inputs), Shape({4}), {-1, 2, 3.75f, -2}, "Add");
    graph.setOutput(graph.mul(a, b));
    check(evaluate(graph, inputs), Shape({4}), {-2, 0, 3.375f, -15}, "Mul");
    graph.setOutput(a);
    check(evaluate(graph, inputs), Shape({4}), {-2, 0, 1.5f, 3}, "Passthrough");
    require(inputs.at("A").data() == std::vector<float>{-2, 0, 1.5f, 3},
            "Executor modified input A");
    require(inputs.at("B").data() == std::vector<float>{1, 2, 2.25f, -5},
            "Executor modified input B");
}

void testExpression1D() {
    using namespace miniai;
    Graph graph;
    const auto a = graph.input("A", Shape({4}));
    const auto b = graph.input("B", Shape({4}));
    const auto c = graph.input("C", Shape({4}));
    graph.setOutput(graph.mul(graph.relu(graph.add(a, b)), c));
    InputMap inputs{
        {"A", Tensor(Shape({4}), {-1, -2, 3, 4})},
        {"B", Tensor(Shape({4}), {2, 1, -5, 1})},
        {"C", Tensor(Shape({4}), {2, 3, 4, -1})}};
    const Tensor actual = evaluate(graph, inputs);
    check(actual, Shape({4}), {2, 0, 0, -5}, "Expression1D");
    std::cout << "Expression1D:";
    for (float value : actual.data()) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';
}

void testExpression2D() {
    using namespace miniai;
    Graph graph;
    const auto a = graph.input("A", Shape({2, 3}));
    const auto b = graph.input("B", Shape({2, 3}));
    const auto c = graph.input("C", Shape({2, 3}));
    graph.setOutput(graph.mul(graph.relu(graph.add(a, b)), c));
    InputMap inputs{
        {"A", Tensor(Shape({2, 3}), {-1, 2, 3, -4, 5, -6})},
        {"B", Tensor(Shape({2, 3}), {3, -3, 1, 4, -2, 7})},
        {"C", Tensor(Shape({2, 3}), {2, 3, -1, 4, 0, 2})}};
    check(evaluate(graph, inputs), Shape({2, 3}), {4, 0, -4, 0, 0, 2},
          "Expression2D");
}

void testSharedIntermediate() {
    using namespace miniai;
    Graph graph;
    const auto a = graph.input("A", Shape({2}));
    const auto b = graph.input("B", Shape({2}));
    const auto c = graph.input("C", Shape({2}));
    const auto sum = graph.add(a, b);
    const auto left = graph.relu(sum);
    const auto right = graph.mul(sum, c);
    graph.setOutput(graph.add(left, right));
    InputMap inputs{
        {"A", Tensor(Shape({2}), {1, -2})},
        {"B", Tensor(Shape({2}), {2, -1})},
        {"C", Tensor(Shape({2}), {2, 4})}};
    check(evaluate(graph, inputs), Shape({2}), {9, -12}, "SharedIntermediate");
}

void testReLUSpecialValues() {
    using namespace miniai;
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Graph graph;
    const auto x = graph.input("X", Shape({7}));
    graph.setOutput(graph.relu(x));
    InputMap inputs{
        {"X", Tensor(Shape({7}), {-inf, -2, -0.0f, 0.0f, 2, inf, nan})}};
    check(evaluate(graph, inputs), Shape({7}), {0, 0, 0, 0, 2, inf, nan},
          "ReLUSpecialValues");
    require(std::signbit(inputs.at("X").data()[2]),
            "Executor modified negative zero input");
}

void testUnusedBranch() {
    using namespace miniai;
    Graph graph;
    const auto a = graph.input("A", Shape({2}));
    const auto unused = graph.input("Unused", Shape({7}));
    (void)graph.mul(unused, unused);
    graph.setOutput(a);
    check(evaluate(graph, {{"A", Tensor(Shape({2}), {2, 3})}}),
          Shape({2}), {2, 3}, "UnusedBranch");
}

void testErrors() {
    using namespace miniai;
    expectThrows<std::invalid_argument>(
        [] { (void)Tensor(Shape({2, 3}), {1, 2}); },
        "Tensor accepted wrong data length");
    expectThrows<std::invalid_argument>(
        [] { (void)Tensor(Shape({2}), {}); },
        "Tensor accepted empty data");

    Graph graph;
    const auto a = graph.input("A", Shape({4}));
    const auto b = graph.input("B", Shape({4}));
    const auto sum = graph.add(a, b);
    expectThrows<std::logic_error>(
        [&] { (void)evaluate(graph, {}); },
        "Executor accepted missing graph output");
    graph.setOutput(sum);
    expectThrows<std::invalid_argument>(
        [&] { (void)evaluate(graph, {}); },
        "Executor accepted missing inputs");
    expectThrows<std::invalid_argument>(
        [&] {
            (void)evaluate(graph, {{"A", Tensor(Shape({4}), {1, 2, 3, 4})}});
        },
        "Executor accepted missing B");
    expectThrows<std::invalid_argument>(
        [&] {
            (void)evaluate(graph, {
                {"A", Tensor(Shape({2, 2}), {1, 2, 3, 4})},
                {"B", Tensor(Shape({4}), {1, 2, 3, 4})}});
        },
        "Executor accepted equal counts but different Shapes");
}

} // namespace

int main() {
    try {
        testAddMulAndPassthrough();
        testExpression1D();
        testExpression2D();
        testSharedIntermediate();
        testReLUSpecialValues();
        testUnusedBranch();
        testErrors();
        std::cout << "All reference executor tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
