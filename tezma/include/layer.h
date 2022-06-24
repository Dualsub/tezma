#pragma once

#include "function.h"

namespace tz
{
    using DType = float;

    class Layer : public Function<DType>
    {
    public:
    private:
        float m_learning_rate;

    public:
        virtual Tensor<DType> forward(const Tensor<DType> &t) const override
        {
            return t;
        }

        virtual Tensor<DType> backward(const Tensor<DType> &t) const override
        {
            return t;
        }
    };

    class LinearLayer : public Function<DType>
    {
    public:
        LinearLayer(float learning_rate = 0.01f) : m_learning_rate(learning_rate) {}

    private:
        DType m_learning_rate;

    public:
        virtual Tensor<DType> forward(const Tensor<DType> &t) const override
        {
            return t;
        }

        virtual Tensor<DType> backward(const Tensor<DType> &t) const override
        {
            return t;
        }
    };
}