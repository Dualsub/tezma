#pragma once

#include "tensor/tensor_base.h"
#include "cuda_ops_impl.h"

namespace tz::cuda
{
    Tensor<float> matmul(const Tensor<float>& t1, const Tensor<float>& t2);
}