#pragma once

#include <vector>
#include <memory>
#include "tezma.h"
#include "layer.h"
#include "function.h"
#include "loss.h"

namespace tz
{
    class Network
    {
    private:
        std::vector<Layer> m_layers;
        std::vector<std::unique_ptr<Function<DType>>> m_activations;
        std::unique_ptr<LossFunction<DType>> m_loss_function;

        Tensor<DType> predict(const Tensor<DType>& inputs);
        void train(const Tensor<DType>& inputs, const Tensor<DType>& outputs, const LossFunction<DType>& loss_func, float learning_rate, size_t epochs);
        Tensor<DType> back_prop(const Tensor<DType>& output, const Tensor<DType>& output_pred, const LossFunction<DType>& loss_func, float learning_rate);

        Tensor<DType> Network::forward(const Tensor<DType>& inputs);
        Tensor<DType> Network::backward(const Tensor<DType>& grad);
    public:

        Network();
        ~Network();

        void add_layer(const Layer &layer, const Function<DType> &activation_function);
    };
}
