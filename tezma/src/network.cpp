#include "network.h"

namespace tz
{
    void Network::add_layer(const Layer &layer, const Function<DType> &activation_function)
    {
        m_layers.push_back(layer);
        m_activations.push_back(std::make_unique<Function<DType>>(activation_function));
    }

    void Network::train(const Tensor<DType>& inputs, const Tensor<DType>& outputs, const LossFunction<DType>& loss_func, float learning_rate, size_t epochs)
    {
        for (size_t e = 0; e < epochs; e++)
        {
            TZ_ASSERT(inputs.shape(0) == outputs.shape(0) && "Not same first shape of data.");
            
            float err = 0;
            for (size_t i = 0; i < inputs.shape(0); i++)
            {
                Tensor<DType> input = inputs.slice({ i });
                Tensor<DType> output = outputs.slice({ i });

                Tensor<DType> pred_output = predict(input);

                err += loss_func.forward(output, pred_output);

                back_prop(output, pred_output, loss_func, learning_rate);
            }

            err /= inputs.shape(0);
        }
    }

    Tensor<DType> Network::forward(const Tensor<DType>& inputs)
    {
        TZ_ASSERT(m_layers.size() == m_activations.size() && "Every layer needs to have and activation function.");

        Tensor<DType> x = inputs;
        for (size_t i = 0; i < m_layers.size(); i++)
        {
            x = m_layers[i].forward(x);
            x = m_activations[i]->forward(x);
        }
        
        return x;
    }

    Tensor<DType> Network::backward(const Tensor<DType>& grad)
    {
        TZ_ASSERT(m_layers.size() == m_activations.size() && "Every layer needs to have and activation function.");

        Tensor<DType> x = grad;
        for (size_t i = m_layers.size() - 1; i >= 0; i--)
        {
            x = m_activations[i]->backward(x);
            x = m_layers[i].backward(x);
        }
        
        return x;
    }
}