cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_CXX_STANDARD 17)

# Cuda support variable
option(CUDA_ENABLED "Cuda support." OFF)

# Cuda support is optional
if(CUDA_ENABLED)
    add_subdirectory(cuda)
    add_subdirectory(examples/cuda_testing)
endif()

add_subdirectory(tezma)
add_subdirectory(examples/xor)
add_subdirectory(examples/mnist)

enable_testing()
add_subdirectory(tests)