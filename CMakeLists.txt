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

add_executable(graph_add_test tests/graph_add_test.cpp)
target_link_libraries(graph_add_test PRIVATE miniai_graph)
add_test(NAME GraphAdd COMMAND graph_add_test)

add_executable(graph_elementwise_test tests/graph_elementwise_test.cpp)
target_link_libraries(graph_elementwise_test PRIVATE miniai_graph)
add_test(NAME GraphElementwise COMMAND graph_elementwise_test)

add_library(miniai_reference lib/Runtime/ReferenceExecutor.cpp)
target_link_libraries(miniai_reference PUBLIC miniai_graph)

add_executable(reference_executor_test tests/reference_executor_test.cpp)
target_link_libraries(reference_executor_test PRIVATE miniai_reference)
add_test(NAME ReferenceExecutor COMMAND reference_executor_test)
