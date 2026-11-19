#include <array>
#include <iostream>

extern "C" float add(float a, float b);

struct TestCase {
    float a;
    float b;
    float expected;
};

int main() {
    const std::array<TestCase, 5> tests{{
        {2.0f, 3.0f, 5.0f},
        {-2.0f, 3.0f, 1.0f},
        {-2.0f, -3.0f, -5.0f},
        {0.0f, 0.0f, 0.0f},
        {1.5f, 2.25f, 3.75f},
    }};

    for (const auto &test : tests) {
        const float actual = add(test.a, test.b);

        if (actual != test.expected) {
            std::cerr << "FAIL: " << test.a << " + " << test.b
                      << ", expected " << test.expected
                      << ", got " << actual << '\n';
            return 1;
        }

        std::cout << test.a << " + " << test.b
                  << " = " << actual << '\n';
    }

    std::cout << "All " << tests.size() << " tests passed\n";
    return 0;
}
