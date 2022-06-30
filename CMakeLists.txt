cmake_minimum_required(VERSION 3.0.0)

add_subdirectory(tezma)
add_subdirectory(examples/mnist)
add_subdirectory(examples/xor)

enable_testing()
add_subdirectory(tests)