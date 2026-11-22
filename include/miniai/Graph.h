#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace miniai {

class Shape {
public:
    explicit Shape(std::vector<std::int64_t> dimensions);

    const std::vector<std::int64_t> &dimensions() const {
        return dimensions_;
    }

    std::size_t rank() const { return dimensions_.size(); }
    std::size_t numElements() const { return numElements_; }

private:
    std::vector<std::int64_t> dimensions_;
    std::size_t numElements_ = 1;
};

using NodeId = std::size_t;

enum class NodeKind {
    Input,
    Add,
    Mul,
    ReLU
};

struct Node {
    NodeId id;
    NodeKind kind;
    std::string name;
    Shape shape;
    std::vector<NodeId> inputs;
};

class Graph {
public:
    NodeId input(std::string name, Shape shape);
    NodeId add(NodeId lhs, NodeId rhs);
    NodeId mul(NodeId lhs, NodeId rhs);
    NodeId relu(NodeId input);

    void setOutput(NodeId id);
    bool hasOutput() const { return output_.has_value(); }
    NodeId output() const;

    const Node &node(NodeId id) const;
    std::size_t nodeCount() const { return nodes_.size(); }

private:
    NodeId binary(NodeKind kind, NodeId lhs, NodeId rhs);

    std::vector<Node> nodes_;
    std::optional<NodeId> output_;
};

} // namespace miniai
