cmake_minimum_required(VERSION 3.0.0)
project(tezma VERSION 0.1.0)

include(CTest)
enable_testing()

add_library(tezma 
src/tezma.cpp
src/net.cpp
src/tensor/tensor_base.cpp
# src/cuda/cuda_ops.cpp
)
target_include_directories(tezma PUBLIC include)

if(CUDA_ENABLED)
    target_link_libraries(tezma PUBLIC cuda_ops)
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
