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
template <typename Func>
size_t matmul_benchmark(size_t n, size_t mat_size, const Func &&function)
{
    std::chrono::high_resolution_clock c;
    size_t acc_ms = 0;
    for (size_t i = 0; i < n; i++)
    {
        auto t1 = c.now();
        tz::Tensor<float> result = function();
        auto t2 = c.now();
        acc_ms += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    }
    return acc_ms / n;
}

int main(int argc, char const *argv[])
{
    tz::seed(0);
    const int mat_size = 64;
    tz::Tensor<float> a = tz::randn<float>({mat_size, mat_size / 2});
    tz::Tensor<float> b = tz::randn<float>({mat_size/2, mat_size});

    std::cout << "Running tiled." << std::endl;
    size_t ms_tiled = matmul_benchmark(1000, mat_size, [&a, &b]()
                                       { return tz::matmul_tiled(a, b, 8); });
    std::cout << "Running conv." << std::endl;
    size_t ms_conv = matmul_benchmark(1000, mat_size, [&a, &b]()
                                      { return tz::matmul(a, b); });

    std::cout << "Tiled:\t" << (mat_size * mat_size * 2 * mat_size) / (float(ms_tiled) * 1e3) << " GFLOP/S" << std::endl;
    std::cout << "Conv:\t"  << (mat_size * mat_size * 2 * mat_size) / (float(ms_conv)  * 1e3) << " GFLOP/S" << std::endl;

    TZ_ASSERT(tz::matmul_tiled(a, b) == tz::matmul(a, b, 4));

    return 0;
}
