cmake_minimum_required(VERSION 3.20)

project(MiniAICompiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(miniai_graph lib/Graph/Graph.cpp)
target_include_directories(miniai_graph PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_testing()

add_executable(graph_test tests/graph_test.cpp)
target_link_libraries(graph_test PRIVATE miniai_graph)

add_test(NAME GraphBasics COMMAND graph_test)
