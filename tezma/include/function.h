#pragma once

#include <memory>
#include "tezma.h"
#include "tensor.h"

namespace tz
{
    template <typename Numeric>
    class Parameter
    {
        Tensor<Numeric>& m_value;
        Tensor<Numeric>& m_grad;

    public:
        Parameter(Tensor<Numeric>& value, Tensor<Numeric>& grad)
            : m_value(value), m_grad(grad)
        {
        }
        Parameter(Tensor<Numeric>&&, Tensor<Numeric>&&) = delete;
        Parameter() = default;
        ~Parameter() = default;

        Tensor<Numeric>& value()
        {
            return m_value;
        }

        const Tensor<Numeric>& grad() const
        {
            return m_grad;
        }
    };

    template <typename Numeric>
    class Function
    {
    private:
    public:
        Function() = default;
        ~Function() = default;

        virtual Tensor<Numeric> forward(const Tensor<Numeric> &t) = 0;
        virtual Tensor<Numeric> backward(const Tensor<Numeric> &t) = 0;
        virtual const std::string& type()
        {
            static std::string s = "Function";
            return s;
        }
        
        virtual std::vector<Parameter<Numeric>> params()
        {
            return std::vector<Parameter<Numeric>>();
        }
    };

    template <typename Numeric>
    using FunctionPtr = std::unique_ptr<Function<Numeric>>;
}