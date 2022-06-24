cmake_minimum_required(VERSION 3.0.0)

add_subdirectory(tezma)
add_subdirectory(examples/mnist)

enable_testing()
add_subdirectory(tests)