cmake_minimum_required(VERSION 3.20)

project(MiniAICompilerModuleLesson LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(LLVM 20.1.8 EXACT REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM config: ${LLVM_DIR}")
message(STATUS "MLIR config: ${MLIR_DIR}")

add_definitions(${LLVM_DEFINITIONS})

add_executable(mlir_module_demo main.cpp)

target_include_directories(mlir_module_demo SYSTEM PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(mlir_module_demo PRIVATE
    MLIRIR
    MLIRSupport
)
