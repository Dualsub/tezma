#pragma once

#include <vector>
#include <memory>
#include "tezma.h"
#include "tensor.h"
#include "layer.h"
#include "function.h"
#include "optimizer.h"
#include "loss.h"

namespace tz
{
    class Net
    {
    private:
        std::vector<FunctionPtr<DType>> m_layers;
        std::vector<Parameter<DType>> m_parameters;

    public:

        Net() = default;
        ~Net() = default;

        void fit(Tensor<DType> &inputs, Tensor<DType> &outputs, LossFunction<DType> &loss_func, Optimizer<DType> &optimizer, float learning_rate, size_t epochs, size_t batch_size = 1);
        void back_prop(const Tensor<DType>& output, const Tensor<DType>& output_pred, LossFunction<DType>& loss_func);
        float eval(Tensor<DType>& inputs, Tensor<DType>& outputs, LossFunction<DType>& loss_func);


        Tensor<DType> backward(const Tensor<DType>& grad);
        Tensor<DType> predict(const Tensor<DType>& inputs);

        // Overload call operator to add layers.
        Tensor<DType> operator()(const Tensor<DType>& input)
        {
            return predict(input);
        }

        template<typename FunctionType, typename... Args>
        void add(Args... args)
        {
            m_layers.push_back(std::make_unique<FunctionType>(args...));
        }
    };
}
