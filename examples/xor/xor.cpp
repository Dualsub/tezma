#include "tezma.h"
#include "layer.h"
#include "loss.h"
#include "tensor.h"
#include "activation.h"
#include <iostream>

int main(int argc, char const *argv[])
{
    tz::seed(0);
    std::cout << "Running XOR Example." << std::endl;

    tz::Tensor<float> inputs = tz::tensor<float>({
        {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}
    });
    inputs.reshape({ 4, 2, 1 });
    
    tz::Tensor<float> outputs = tz::tensor<float>({
        0.0f, 1.0f, 1.0f, 0.0f
    });
    outputs.reshape({ 4, 1, 1 });

    auto a1 = tz::Tanh();
    auto a2 = tz::Tanh();
    auto l1 = tz::LinearLayer(2, 3);
    auto l2 = tz::LinearLayer(3, 1);
    auto loss = tz::MSELoss();
    float error = 0.0;

    const size_t epochs = 10000;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for(size_t i = 0; i < inputs.shape(0); i++)
        {
            auto x = inputs.slice({i});
            auto y = outputs.slice({i});
            x = l1.forward(x);
            x = a1.forward(x);
            x = l2.forward(x);
            x = a2.forward(x);

            error += loss.forward(x, y);
            auto grad = loss.backward(x, y);
        
            grad = a2.backward(grad);
            grad = l2.backward(grad);
            grad = a1.backward(grad);
            grad = l1.backward(grad);
        }

        error /= inputs.shape(0);

        std::cout << epoch+1 << "/" << epochs << " Error: " << error << std::endl;
    }

    // Print model results.
    for(size_t i = 0; i < inputs.shape(0); i++)
    {
        auto input = inputs.slice({i});
        auto output = outputs.slice({i});
        auto x = input;
        x = l1.forward(x);
        x = a1.forward(x);
        x = l2.forward(x);
        x = a2.forward(x);
        std::cout << (int)std::round(input[0]) << ", " << (int)std::round(input[1]) << " -> " << (int)std::round(x[0]) << std::endl;
    }

    // // Creating model
    // size_t epochs = 10000;
    
    // tz::Net<float> net;

    // net.add_layer(LinearLayer({2, 3}), Tanh());
    // net.add_layer(LinearLayer({3, 1}), Tanh());

    // net.train(inputs, outputs, MSELoss(), 0.1, epochs);

    // net.saveas("nets/xor_net.tzn");

    return 0;
}