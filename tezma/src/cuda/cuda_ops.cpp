#include "cuda/cuda_ops.h"

namespace tz::cuda
{
    Tensor<float> matmul(const Tensor<float>& t1, const Tensor<float>& t2)
    {
        CudaData A(t1.data(), t1.size());
        CudaData B(t2.data(), t2.size());
        CudaData C(t1.shape(0) * t2.shape(1));

        cuda_matmul_impl(A.data(), B.data(), C.data(), t1.shape(0), t1.shape(1), t2.shape(1));
        
        Tensor<float> t3(C.to_host(), { t1.shape(0), t2.shape(1) });

        return t3;
    }
} 
