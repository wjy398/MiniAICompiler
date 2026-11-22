#pragma once

#include "miniai/Graph.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace miniai {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "MiniAICompiler requires IEEE 754 float32");

// Owns contiguous, row-major float32 data. Shape alone never allocates data.
class Tensor {
public:
    Tensor(Shape shape, std::vector<float> data)
        : shape_(std::move(shape)), data_(std::move(data)) {
        if (data_.size() != shape_.numElements()) {
            throw std::invalid_argument(
                "Tensor data length must match Shape element count");
        }
    }

    const Shape &shape() const { return shape_; }
    const std::vector<float> &data() const { return data_; }

private:
    Shape shape_;
    std::vector<float> data_;
};

} // namespace miniai
