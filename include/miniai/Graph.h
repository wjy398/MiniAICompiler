#pragma once

#include <cstddef>
#include <cstdint>
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
    Input
};

struct Node {
    NodeId id;
    NodeKind kind;
    std::string name;
    Shape shape;
};

class Graph {
public:
    NodeId input(std::string name, Shape shape);
    const Node &node(NodeId id) const;
    std::size_t nodeCount() const { return nodes_.size(); }

private:
    std::vector<Node> nodes_;
};

} // namespace miniai
