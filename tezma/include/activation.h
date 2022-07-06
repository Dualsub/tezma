#pragma once

#include "function.h"

namespace tz
{
    using DType = float;

    class Tanh : public Function<DType>
    {
    private:
        Tensor<DType> m_inputs;
    public:
        Tanh() = default;
        ~Tanh() = default;


        Tensor<DType> forward(const Tensor<DType> &t) override
        {
            m_inputs = t;
            Tensor<DType> result(t.shape());
            for (size_t i = 0; i < t.size(); i++)
            {
                result[i] = (DType)std::tanh(t[i]);
            }
            return result;
        }

        Tensor<DType> backward(const Tensor<DType> &t) override
        {
            Tensor<DType> result(m_inputs.shape());
            for (size_t i = 0; i < m_inputs.size(); i++)
            {
                result[i] = 1.0f - (DType)std::pow(std::tanh(m_inputs[i]), 2);
            }

            result *= t;

            return result;
        }
    };

    
    class ReLU : public Function<DType>
    {
    private:
        Tensor<DType> m_inputs;
    public:
        ReLU() = default;
        ~ReLU() = default;


        Tensor<DType> forward(const Tensor<DType> &t) override
        {
            m_inputs = t;
            Tensor<DType> result(t.shape());
            for (size_t i = 0; i < t.size(); i++)
            {
                result[i] = (t[i] > 0) ? t[i] : 0;
            }
            return result;
        }

        Tensor<DType> backward(const Tensor<DType> &t) override
        {
            Tensor<DType> result(m_inputs.shape());
            for (size_t i = 0; i < m_inputs.size(); i++)
            {
                result[i] = (m_inputs[i] > 0) ? (DType)1 : (DType)0;
            }

            result *= t;

            return result;
        }
    };

}