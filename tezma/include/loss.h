#pragma once

#include "tezma.h"
#include "tensor.h"

namespace tz
{
    using DType = float;

    template<typename Numeric>
    class LossFunction
    {
    private:
        
    public:
        LossFunction() = default;
        ~LossFunction() = default;

        virtual Numeric forward(const Tensor<Numeric>& y_true, const Tensor<Numeric>& y_pred) = 0;
        virtual Tensor<Numeric> backward(const Tensor<Numeric>& y_true, const Tensor<Numeric>& y_pred) = 0;

    };

    class MSELoss : public LossFunction<DType>
    {
    private:
        
    public:
        MSELoss() = default;
        ~MSELoss() = default;

        virtual DType forward(const Tensor<DType>& y_pred, const Tensor<DType>& y_true) override
        {
            return mean(pow(y_true - y_pred, 2));
        }

        virtual Tensor<DType> backward(const Tensor<DType>& y_pred, const Tensor<DType>& y_true) override
        {
            DType factor = (2.0f/y_true.size());
            auto diff = y_pred - y_true;
            return  diff * factor;
        }

    };


}