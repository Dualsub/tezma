#pragma once

#include "tensor.h"

namespace tz
{
    template <typename Numeric>
    class Function
    {
    private:
    public:
        Function() = default;
        ~Function() = default;

        virtual Tensor<Numeric> forward(const Tensor<Numeric> &t) const = 0;
        virtual Tensor<Numeric> backward(const Tensor<Numeric> &t) const = 0;
    };
}