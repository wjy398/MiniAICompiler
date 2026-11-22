cmake_minimum_required(VERSION 3.20)

project(MiniAICompiler LANGUAGES C CXX)

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

set(LLVM_DIR "/usr/lib/llvm-20/lib/cmake/llvm" CACHE PATH "LLVM CMake config")
set(MLIR_DIR "/usr/lib/llvm-20/lib/cmake/mlir" CACHE PATH "MLIR CMake config")
find_package(LLVM 20.1.8 EXACT REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})

add_library(miniai_irgen lib/IRGen/IRGen.cpp)
target_include_directories(miniai_irgen SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(miniai_irgen PUBLIC
    miniai_graph
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRBufferizationDialect
    MLIRMemRefDialect
    MLIRIR
    MLIRSupport
)

add_executable(miniai-ir tools/miniai-ir.cpp)
target_link_libraries(miniai-ir PRIVATE miniai_irgen)

add_executable(irgen_test tests/irgen_test.cpp)
target_link_libraries(irgen_test PRIVATE miniai_irgen MLIRParser)
add_test(NAME IRGeneration COMMAND irgen_test)

add_library(miniai_optimizations lib/Transforms/ElementwiseFusion.cpp)
target_include_directories(miniai_optimizations PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(miniai_optimizations SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(miniai_optimizations PUBLIC
    MLIRLinalgTransforms MLIRTransforms MLIRPass
    MLIRLinalgDialect MLIRArithDialect MLIRFuncDialect MLIRIR MLIRSupport)
target_link_libraries(miniai-ir PRIVATE miniai_optimizations)
