#pragma once

#include "tensor.h"

namespace tz
{
    template <typename Numeric>
    class Parameter
    {
        Tensor<Numeric> m_value;
        Tensor<Numeric> m_grad;

    public:
        Parameter(const Tensor<Numeric>& value, const Tensor<Numeric>& grad)
            : m_value(value), m_grad(grad)
        {
        }

        const Tensor<Numeric>& value() const
        {
            return m_value;
        }

        Tensor<Numeric>& value()
        {
            return m_value;
        }

        const Tensor<Numeric>& grad() const
        {
            return m_grad;
        }

        Tensor<Numeric>& grad()
        {
            return m_grad;
        }
    };
};