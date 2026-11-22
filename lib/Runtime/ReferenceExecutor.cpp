#include "miniai/ReferenceExecutor.h"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace miniai {
namespace {

const Tensor &getValue(const std::vector<std::optional<Tensor>> &values,
                       NodeId id) {
    const auto &value = values.at(id);
    if (!value) {
        throw std::logic_error("Graph dependency has not been evaluated");
    }
    return *value;
}

} // namespace

Tensor evaluate(const Graph &graph, const InputMap &inputs) {
    const NodeId output = graph.output();

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

    // Graph construction guarantees that operands precede their consumers.
    // Each reachable node is evaluated once; shared results are reused.
    std::vector<std::optional<Tensor>> values(graph.nodeCount());
    for (NodeId id = 0; id < graph.nodeCount(); ++id) {
        if (!required[id]) {
            continue;
        }
        const Node &node = graph.node(id);
        if (node.kind == NodeKind::Input) {
            const auto found = inputs.find(node.name);
            if (found == inputs.end()) {
                throw std::invalid_argument(
                    "Missing tensor for input: " + node.name);
            }
            if (found->second.shape().dimensions() != node.shape.dimensions()) {
                throw std::invalid_argument(
                    "Tensor Shape does not match input: " + node.name);
            }
            values[id].emplace(found->second);
            continue;
        }

        const Tensor &lhs = getValue(values, node.inputs.at(0));
        std::vector<float> result(lhs.data().size());
        switch (node.kind) {
        case NodeKind::Add:
        case NodeKind::Mul: {
            const Tensor &rhs = getValue(values, node.inputs.at(1));
            for (std::size_t i = 0; i < result.size(); ++i) {
                result[i] = node.kind == NodeKind::Add
                    ? lhs.data()[i] + rhs.data()[i]
                    : lhs.data()[i] * rhs.data()[i];
            }
            break;
        }
        case NodeKind::ReLU:
            for (std::size_t i = 0; i < result.size(); ++i) {
                const float value = lhs.data()[i];
                result[i] = std::isnan(value)
                    ? value : (value > 0.0f ? value : 0.0f);
            }
            break;
        default:
            throw std::logic_error("Unsupported graph operation");
        }
        values[id].emplace(node.shape, std::move(result));
    }

    if (!values.at(output)) {
        throw std::logic_error("Graph output has not been evaluated");
    }
    return std::move(*values[output]);
}

} // namespace miniai
