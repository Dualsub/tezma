#include "net.h"
#include "optimizer.h"
#include <chrono>

namespace tz
{
    /**
     * > This function takes in a set of inputs and outputs, a loss function, an optimizer, a learning
     * rate, the number of epochs, and the batch size, and then trains the network
     * 
     * @param inputs The input data.
     * @param outputs The expected output of the network.
     * @param loss_func The loss function to use.
     * @param optimizer The optimizer to use.
     * @param learning_rate The learning rate of the optimizer.
     * @param epochs The number of times the network will see the entire dataset.
     * @param batch_size The number of samples to use in each batch.
     */
    void Net::fit(Tensor<DType> &inputs, Tensor<DType> &outputs, LossFunction<DType> &loss_func, Optimizer<DType> &optimizer, float learning_rate, size_t epochs, size_t batch_size)
    {
        TZ_ASSERT(inputs.shape(0) == outputs.shape(0) && "Not same first shape of data.");

        optimizer.set_learning_rate(learning_rate);

        // Time the training.
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t e = 0; e < epochs; e++)
        {
            size_t num_iter = inputs.shape(0) / batch_size;

            auto epoch_start = std::chrono::high_resolution_clock::now();

            float err = 0;
            for (size_t i = 0; i < num_iter; i++)
            {
                m_parameters.clear();

                Tensor<DType> input = inputs[Range(i, i + batch_size)];
                Tensor<DType> output = outputs[Range(i, i + batch_size)];
                input.reshape({batch_size, input.shape(1)});
                output.reshape({batch_size, output.shape(1)});

                Tensor<DType> pred_output = predict(input);

                err += loss_func.forward(output, pred_output);
                back_prop(output, pred_output, loss_func);

                optimizer.step(m_parameters);
            }

            auto epoch_end = std::chrono::high_resolution_clock::now();

            err /= inputs.shape(0);
            std::cout << "Epoch: " << e+1 << " Error: " << err << " Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start).count() << " ms" << std::endl;
        }

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Training time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
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

            for (auto &param : m_layers[i]->params())
            {
                m_parameters.push_back(param);
            }
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
            Tensor<DType> x = inputs.slice({i});
            Tensor<DType> y = outputs.slice({i});
            x.reshape({1, x.shape(0)});
            y.reshape({1, y.shape(0)});

            Tensor<DType> pred_y = predict(x);

            err += loss_func.forward(y, pred_y);
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