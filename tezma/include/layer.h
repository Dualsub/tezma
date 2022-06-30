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
        virtual Tensor<DType> forward(const Tensor<DType> &t) override
        {
            return t;
        }

        virtual Tensor<DType> backward(const Tensor<DType> &t) override
        {
            return t;
        }
    };

    class LinearLayer : public Function<DType>
    {
    public:
        LinearLayer(size_t input_size, size_t output_size, float learning_rate = 0.01f)
            : m_learning_rate(learning_rate)
        {
            m_weights = tz::randn<DType>({output_size, input_size});
            m_bias = tz::randn<DType>({output_size, 1});
        }

    private:
        float m_learning_rate;
        Tensor<DType> m_input;
        Tensor<DType> m_weights;
        Tensor<DType> m_bias;
    public:
        virtual Tensor<DType> forward(const Tensor<DType> &t) override
        {
            m_input = t;
            return matmul(m_weights, t) + m_bias;
        }

        virtual Tensor<DType> backward(const Tensor<DType> &t) override
        {
            auto weights_gradient = matmul(t, transpose(m_input));
            auto input_gradient = matmul(transpose(m_weights), t);
            m_weights = m_weights - (m_learning_rate * weights_gradient);
            m_bias = m_bias - (m_learning_rate * t);

            return input_gradient;
        }

        const Tensor<DType>& weights() const
        {
            return m_weights;
        }

        const Tensor<DType>& bias() const
        {
            return m_bias;
        }
    };
}