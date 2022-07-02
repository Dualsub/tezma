#include "net.h"

namespace tz
{
    /**
     * > This function takes in a set of inputs and outputs, a loss function, a learning rate, and the
     * number of epochs to train for. It then loops through the inputs and outputs, calculates the
     * loss, and backpropagates the error
     * 
     * @param inputs The input data.
     * @param outputs The expected output of the network.
     * @param loss_func The loss function to use.
     * @param learning_rate The learning rate of the network.
     * @param epochs The number of times the network will be trained on the data.
     */
    void Net::fit(Tensor<DType> &inputs, Tensor<DType> &outputs, LossFunction<DType> &loss_func, float learning_rate, size_t epochs)
    {
        for (size_t e = 0; e < epochs; e++)
        {
            TZ_ASSERT(inputs.shape(0) == outputs.shape(0) && "Not same first shape of data.");

            float err = 0;
            for (size_t i = 0; i < inputs.shape(0); i++)
            {
                Tensor<DType> input = inputs.slice({i});
                Tensor<DType> output = outputs.slice({i});

                Tensor<DType> pred_output = predict(input);

                err += loss_func.forward(output, pred_output);
                back_prop(output, pred_output, loss_func);
            }

            err /= inputs.shape(0);
            std::cout << "Epoch: " << e << " Error: " << err << std::endl;
        }
    }

    /**
     * > The `predict` function takes in a tensor of inputs and returns a tensor of outputs
     * 
     * @param inputs The input data.
     * 
     * @return The output of the last layer.
     */
    Tensor<DType> Net::predict(const Tensor<DType> &inputs)
    {
        Tensor<DType> x = inputs;
        for (size_t i = 0; i < m_layers.size(); i++)
        {
            x = m_layers[i]->forward(x);
        }

        return x;
    }

    /**
     * > The function takes the output of the network and the expected output, and then calculates the
     * gradient of the loss function with respect to the weights of the network
     * 
     * @param output the actual output of the network
     * @param output_pred the output of the network
     * @param loss_func The loss function to use.
     */
    void Net::back_prop(const Tensor<DType> &output, const Tensor<DType> &output_pred, LossFunction<DType> &loss_func)
    {
        auto grad = loss_func.backward(output, output_pred);
        for (size_t i = m_layers.size() - 1; i > 0; i--)
        {
            grad = m_layers[i]->backward(grad);
        }
    }

    /**
     * > This function takes in a batch of inputs and outputs, and returns the average loss of the
     * batch
     * 
     * @param inputs the input data
     * @param outputs the expected output of the network
     * @param loss_func The loss function to use.
     * 
     * @return The average error of the network.
     */
    float Net::eval(Tensor<DType> &inputs, Tensor<DType> &outputs, LossFunction<DType> &loss_func)
    {
        TZ_ASSERT(inputs.shape(0) == outputs.shape(0) && "Not same first shape of data.");

        float err = 0;
        for (size_t i = 0; i < inputs.shape(0); i++)
        {
            Tensor<DType> input = inputs.slice({i});
            Tensor<DType> output = outputs.slice({i});

            Tensor<DType> pred_output = predict(input);

            err += loss_func.forward(output, pred_output);
        }

        err /= inputs.shape(0);
        return err;
    }

    Tensor<DType> Net::backward(const Tensor<DType> &grad)
    {
        Tensor<DType> x = grad;
        for (size_t i = m_layers.size() - 1; i >= 0; i--)
        {
            x = m_layers[i]->backward(x);
        }

        return x;
    }
}