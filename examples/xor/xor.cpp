#include "net.h"
#include "activation.h"
#include "layer.h"
#include "serialization.h"
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

    tz::Net net;

    net.add<tz::LinearLayer>(2, 3);
    net.add<tz::Tanh>();
    net.add<tz::LinearLayer>(3, 1);
    net.add<tz::Tanh>();
    
    // net.fit(inputs, outputs, tz::MSELoss(), 0.01f, 10000);

    // Print the predictions.
    for (size_t i = 0; i < inputs.shape(0); i++)
    {
        tz::Tensor<float> input = inputs.slice({ i });
        tz::Tensor<float> output = net(input);
        std::cout << input[0] << ", " << input[1] << " -> " << (int)std::round(output[0]) << std::endl;
    }

    // net.saveas("nets/xor_net.tzn");
    {
        tz::SerializationContext context;
        tz::serialize(inputs, context);
        context.save("xor_inputs.tzn");
    }

    {
        tz::SerializationContext context;
        context.open("xor_inputs.tzn");
        auto d = tz::deserialize<float>(context);
        std::cout << d << std::endl;
    }


    return 0;
}