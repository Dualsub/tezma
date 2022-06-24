#pragma once

#include "tezma.h"
#include "tensor.h"

namespace tz
{
    template<typename Numeric>
    class LossFunction
    {
    private:
        
    public:
        LossFunction();
        ~LossFunction();

        virtual Numeric forward(const Tensor<Numeric>& y_true, const Tensor<Numeric>& y_pred) const = 0;
        virtual Tensor<Numeric> backward(const Tensor<Numeric>& y_true, const Tensor<Numeric>& y_pred) const = 0;

    };

    template<typename Numeric>
    class MSELoss : public LossFunction<Numeric>
    {
    private:
        
    public:
        MSELoss() = default;
        ~MSELoss() = default;

        virtual Numeric forward(const Tensor<Numeric>& y_true, const Tensor<Numeric>& y_pred) const override
        {
            size_t batch_size = y_true.shape(0);
            Numeric sum_tot = 0.0f;

            for (size_t i = 0; i < batch_size; i++)
            {
                Numeric sum;

                for (size_t i = 0; i < y_true.size(); i++)
                {
                    sum += std::pow(y_true[i] - y_pred[i], 2);
                }

                sum_tot += sum / y_true.size();
            }

            Numeric result = sum_tot / batch_size;
            return result;
        }

        virtual Tensor<Numeric> backward(const Tensor<Numeric>& y_true, const Tensor<Numeric>& y_pred) const override
        {
            Tensor<Numeric> t(y_true.shape());
            if(y_true.shape() != y_pred.shape())
            {
                for (size_t i = 0; i < y_pred.shape(0); i++)
                {
                   t.set_slice({ i }, y_pred.slice({ i }) - y_true);
                }
            }
            else
            {
                t = y_pred - y_true;
            }

            Numeric factor = 2/t.size();
            t *= factor;

            Tensor<Numeric> result = make_tensor({ mean(t) });
            result.reshape({result.shape(0), 1});
            return result;
        }

    };


}