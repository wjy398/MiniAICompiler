#include <array>
#include <cstddef>
#include <iostream>

int main() {
    const std::array<float, 4> a{1, 2, 3, 4};
    const std::array<float, 4> b{10, 20, 30, 40};
    std::array<float, 4> y{};

    for (std::size_t i = 0; i < a.size(); ++i) {
        y[i] = a[i] + b[i];
    }

    for (float value : y) {
        std::cout << value << ' ';
    }
    std::cout << '\n';

    return 0;
}
