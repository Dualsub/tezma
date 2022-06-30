#include "tensor.h"
#include "function.h"
#include "layer.h"
#include <iostream>
#include <chrono>

/**
 * It runs a function `n` times, and returns the average time taken to run the function
 *
 * @param n The number of times to run the benchmark.
 * @param mat_size The size of the matrix to multiply.
 * @param function The function to benchmark.
 *
 * @return The average time taken to perform the matrix multiplication.
 */
// template <typename Func>
// size_t matmul_benchmark(size_t n, size_t mat_size, const Func &&function)
// {
//     std::chrono::high_resolution_clock c;
//     size_t acc_ms = 0;
//     for (size_t i = 0; i < n; i++)
//     {
//         tz::Tensor<float> a = tz::randn<float>({mat_size, mat_size});
//         tz::Tensor<float> b = tz::randn<float>({mat_size, mat_size});
//         auto t1 = c.now();
//         tz::Tensor<float> result = function(a, b);
//         auto t2 = c.now();
//         acc_ms += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//     }
//     return acc_ms / n;
// }

// int main(int argc, char const *argv[])
// {
//     tz::seed(0);
//     size_t ms_conv = matmul_benchmark(1000, 1024, [](auto a, auto b)
//                                       { return tz::matmul(a, b); });
//     size_t ms_tiled = matmul_benchmark(1000, 1024, [](auto a, auto b)
//                                        { return tz::matmul_tiled(a, b); });

//     // std::cout << ms_conv << std::endl;
//     // std::cout << ms_tiled << std::endl;
//     tz::Tensor<float> a = tz::randn<float>({4, 4});
//     tz::Tensor<float> b = tz::randn<float>({4, 1});
//     tz::Tensor<float> c({{3.0f, 2.0f}, {2.0f, 3.0f}});
//     tz::Tensor<float> d({2.0f, 1.0f});
//     d.reshape({2, 1});

//     std::cout << tz::squeeze(tz::randn<float>({4, 4})) << std::endl;
//     return 0;
// }
