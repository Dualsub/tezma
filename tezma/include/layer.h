#pragma once

#include "function.h"
#include "optimizer.h"

namespace tz
{
    #define TZ_LAYER(name) \
    virtual const std::string& type() override { static std::string s = #name; return s; }

    using DType = float;

    class Layer : public Function<DType>
    {
    public:
    private:
        std::vector<Parameter<DType>> m_params;
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
        TZ_LAYER(LinearLayer)

        LinearLayer(size_t input_size, size_t output_size)
        {
            m_weights = tz::randn<DType>({input_size, output_size});
            m_bias = tz::randn<DType>({1, output_size});
        }

    private:
        float m_learning_rate = 0.01f;
        Tensor<DType> m_input;
        Tensor<DType> m_weights;
        Tensor<DType> m_bias;
        Tensor<DType> m_weights_grad;
        Tensor<DType> m_bias_grad;
    public:

        virtual Tensor<DType> forward(const Tensor<DType> &t) override
        {
            m_input = t;
            return dot(t, m_weights) + m_bias;
        }

        virtual Tensor<DType> backward(const Tensor<DType> &t) override
        {
            auto weights_gradient = dot(transpose(m_input), t); 
            auto input_gradient = dot(t, transpose(m_weights));
            
            m_weights_grad = weights_gradient;
            m_bias_grad = t;

            // m_weights = m_weights - (m_learning_rate * weights_gradient);
            // m_bias = m_bias - (m_learning_rate * t);

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

        std::vector<Parameter<DType>> params()
        {
            return {
                Parameter<DType>(m_weights, m_weights_grad),
                Parameter<DType>(m_bias, m_bias_grad)
            };
        }
    };
}