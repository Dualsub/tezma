#pragma once

#include <memory>
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

        virtual Tensor<Numeric> forward(const Tensor<Numeric> &t) = 0;
        virtual Tensor<Numeric> backward(const Tensor<Numeric> &t) = 0;
    };

    template <typename Numeric>
    using FunctionPtr = std::unique_ptr<Function<Numeric>>;
}