#include "miniai/Graph.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace miniai {

Shape::Shape(std::vector<std::int64_t> dimensions)
    : dimensions_(std::move(dimensions)) {
    if (dimensions_.empty() || dimensions_.size() > 2) {
        throw std::invalid_argument("Shape rank must be 1 or 2");
    }

    const auto limit = std::numeric_limits<std::size_t>::max();

    for (std::int64_t dimension : dimensions_) {
        if (dimension <= 0) {
            throw std::invalid_argument(
                "Shape dimensions must be positive");
        }

        if (static_cast<std::uint64_t>(dimension) > limit) {
            throw std::overflow_error("Shape dimension is too large");
        }

        const auto size = static_cast<std::size_t>(dimension);

        if (numElements_ > limit / size) {
            throw std::overflow_error("Shape element count overflow");
        }

        numElements_ *= size;
    }
}

NodeId Graph::input(std::string name, Shape shape) {
    if (name.empty()) {
        throw std::invalid_argument("Input name must not be empty");
    }

    for (const Node &existing : nodes_) {
        if (existing.name == name) {
            throw std::invalid_argument(
                    "Duplicate input name: " + name);
        }
    }

    const NodeId id = nodes_.size();
    nodes_.push_back(
            Node{id, NodeKind::Input, std::move(name), std::move(shape)});
    return id;
}

const Node &Graph::node(NodeId id) const {
    if (id >= nodes_.size()) {
        throw std::out_of_range("Invalid node ID");
    }

    return nodes_[id];
}

} // namespace miniai
