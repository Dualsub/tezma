#pragma once

#include "tensor.h"

namespace tz
{
    template <typename Numeric>
    class Function
    {
    private:
    protected:
        Function() = default;
        ~Function() = default;

        virtual Tensor<Numeric> forward(Tensor<Numeric> &t) = 0;
        virtual Tensor<Numeric> backward(Tensor<Numeric> &t) = 0;
    };

    template <typename Numeric>
    class TestFunction : public Function<Numeric>
    {
    private:
    public:
        TestFunction() = default;
        ~TestFunction() = default;

        virtual Tensor<Numeric> forward(Tensor<Numeric> &t) override
        {
            return t;
        }

        virtual Tensor<Numeric> backward(Tensor<Numeric> &t) override
        {
            return t;
        }
    };

}