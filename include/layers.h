#pragma once

#include "function.h"
namespace tz
{
    class Layer : public Function<Layer::DType>
    {
    public:
        using DType = float;

    private:
        DType m_learning_rate;

    public:
        virtual Tensor<DType> forward(Tensor<DType> &t) override
        {
            return t;
        }

        virtual Tensor<DType> backward(Tensor<DType> &t) override
        {
            return t;
        }
    };

    class LinearLayer : public Function<Layer::DType>
    {
    public:
        using DType = float;

    private:
        DType m_learning_rate;

    public:
        virtual Tensor<DType> forward(Tensor<DType> &t) override
        {
            return t;
        }

        virtual Tensor<DType> backward(Tensor<DType> &t) override
        {
            return t;
        }
    };
}