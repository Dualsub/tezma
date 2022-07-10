#include "tezma.h"
#include "tensor.h"
#include <chrono>

#define NUM_ITERATIONS 10000

int main(int argc, char const *argv[])
{
    {
        auto t1 = tz::randn<float>({ 1, 784 });
        auto t2 = tz::randn<float>({ 784, 784 });

        // Time the operation
        auto start = std::chrono::high_resolution_clock::now();
        CudaData A(t1.data(), t1.size());
        CudaData B(t2.data(), t2.size());
        CudaData C(t1.shape(0) * t2.shape(1));
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Time to allocate: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        assert(t1.shape(1) == t2.shape(0) && "Shape mismatch");
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_ITERATIONS; i++)
        {
            cuda_matmul_impl(A.data(), B.data(), C.data(), (int)t1.shape(0), (int)t1.shape(1), (int)t2.shape(1));
            auto t3_gpu = tz::Tensor<float>(C.to_host(), { 256, 256 });
        }
        end = std::chrono::high_resolution_clock::now();

        std::cout << "GPU time: " << (float)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / NUM_ITERATIONS << " ms" << std::endl;

        // Time the operation
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_ITERATIONS; i++)
        {
            auto t3_cpu = tz::matmul(t1, t2);
        }
        end = std::chrono::high_resolution_clock::now();

        std::cout << "CPU time: " << (float)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / NUM_ITERATIONS << " ms" << std::endl;
    }

    return 0;
}
