#include "tezma.h"
#include "tensor.h"
#include "net.h"
#include "layer.h"
#include "activation.h"
#include "optimizer.h"
#include "serialization.h"

using tz::operator<<;

int main(int argc, char const *argv[])
{
    // tz::seed(0);
    std::cout << "Running MNIST Example." << std::endl;

    // Load MNIST dataset.
    std::cout << "Loading MNIST dataset..." << std::endl;
    tz::Tensor<float> x_train = tz::load<float>("C:/dev/repos/tezma/datasets/mnist/x_train.tzn");
    tz::Tensor<float> y_train = tz::load<float>("C:/dev/repos/tezma/datasets/mnist/y_train.tzn");
    std::cout << "Done. "
              << "Traning set size: " << x_train.shape(0) << std::endl;

    // Create a network with a single hidden layer.
    std::cout << "Creating network." << std::endl;
    tz::Net net;

    net.add<tz::LinearLayer>(28 * 28, 40);
    net.add<tz::Tanh>();
    net.add<tz::LinearLayer>(40, 40);
    net.add<tz::Tanh>();
    net.add<tz::LinearLayer>(40, 40);
    net.add<tz::Tanh>();
    net.add<tz::LinearLayer>(40, 10);
    net.add<tz::Tanh>();

    // Train the network.
    std::cout << "Training network." << std::endl;
    net.fit(x_train, y_train, tz::MSELoss(), tz::SGDOptimizer(), 0.01f, 100);

    tz::Tensor<float> x_test = tz::load<float>("C:/dev/repos/tezma/datasets/mnist/x_test.tzn");
    tz::Tensor<float> y_test = tz::load<float>("C:/dev/repos/tezma/datasets/mnist/y_test.tzn");

    float err = net.eval(x_test, y_test, tz::MSELoss());
    std::cout << "Validation error: " << err << std::endl;

    // Print the predictions.
    size_t count = std::min(x_test.shape(0), size_t(10000));
    size_t num_right = 0;
    for (size_t i = 0; i < count; i++)
    {
        tz::Tensor<float> x = x_test.slice({i});
        tz::Tensor<float> y = y_test.slice({i});
        x.reshape({1, x.shape(0)});
        y.reshape({1, y.shape(0)});
        tz::Tensor<float> y_pred = net(x);
        size_t pred = tz::argmax(y_pred) + 1;
        size_t actual = tz::argmax(y) + 1;
        num_right += pred == actual;
        // std::cout << actual << " -> " << pred << std::endl;
    }

    std::cout << "Accuracy: " << std::round((float(num_right) / count) * 100) << "%" << std::endl;

    return 0;
}
