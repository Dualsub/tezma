#include "tezma.h"
#include "net.h"
#include "layer.h"
#include "activation.h"
#include "serialization.h"

int main(int argc, char const *argv[])
{
    tz::seed(0);
    std::cout << "Running MNIST Example." << std::endl;

    tz::Tensor<float> x_train = tz::load<float>("C:/dev/repos/tezma/datasets/mnist/x_train.tzn");
    tz::Tensor<float> y_train = tz::load<float>("C:/dev/repos/tezma/datasets/mnist/y_train.tzn");

    tz::Net net;

    net.add<tz::LinearLayer>(28 * 28, 40);
    net.add<tz::Tanh>();
    net.add<tz::LinearLayer>(40, 10);
    net.add<tz::Tanh>();
    
    net.fit(x_train, y_train, tz::MSELoss(), 0.01f, 100);

    tz::Tensor<float> x_test = tz::load<float>("C:/dev/repos/tezma/datasets/mnist/x_test.tzn");
    tz::Tensor<float> y_test = tz::load<float>("C:/dev/repos/tezma/datasets/mnist/y_test.tzn");

    float err = net.eval(x_test, y_test, tz::MSELoss());
    std::cout << "Validation error: " << err << std::endl;

    // Print the predictions.
    size_t count = std::min(x_test.shape(0), size_t(10000));
    size_t num_right = 0;
    for (size_t i = 0; i < count; i++)
    {
        tz::Tensor<float> x = x_test.slice({ i });
        tz::Tensor<float> y = y_test.slice({ i });
        tz::Tensor<float> y_pred = net(x);
        size_t pred = tz::argmax(y_pred)+1;
        size_t actual = tz::argmax(y)+1;
        num_right += pred == actual;
        std::cout << actual << " -> " << pred << std::endl;
    }

    std::cout << "Accuracy: " << std::round((float(num_right) / count) * 100) << "%" << std::endl;

    return 0;
}
