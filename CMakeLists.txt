cmake_minimum_required(VERSION 3.26)

project(
    cudakv
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES "90")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CUDA_API_WRAPPERS_DIR ${CMAKE_SOURCE_DIR}/third_party/cuda-api-wrappers)

enable_language(CUDA)

file(GLOB_RECURSE SRC_FILES
    "src/*.cpp"
    "src/*.cu"
)
file(GLOB_RECURSE HEADER_FILES
    "include/*.hpp"
    "include/*.h"
    "include/*.cu"
    "include/*.cuh"
)

find_package(CUDAToolkit REQUIRED)

add_library(cuda_kv STATIC ${SRC_FILES})

target_include_directories(cuda_kv PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CUDA_API_WRAPPERS_DIR}/src
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(cuda_kv
  PRIVATE CUDA::cudart
  PRIVATE CUDA::cuda_driver
)

# target_compile_options(cuda_kv PRIVATE
#   $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native --default-stream per-thread>
# )