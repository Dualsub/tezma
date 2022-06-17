# TEZMA 

## A C++ machine learning framework, created for no reason.

Tezma is a small C++ machine learning framework containing the following features:
- Small tensor math library(mirroring NumPy)
- Layers(Linear, Conv2D)
- Loss(Relu, Tanh)
- Optimizer(Adam)
- Network class(adding layers, traning, prediction)
- Serialization(saving and loading of net)

Tezma was created for learning purposes.

## Usage

To use Tezma in your project, firstly clone the repo like so

```bash
$ git clone https://github.com/Dualsub/tezma.git tezma
$ cd tezma
$ mkdir build
$ cd build
$ cd cmake ..
```

## Examples

### XOR example
XOR is not a linear function and is therefore a good.

```cpp

int main(int argc, char const *argv[])
{
    tz::Tensor<float> inputs = tz::tensor<float>({
        {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}
    });
    inputs.reshape({ 4, 2, 1 });
    
    tz::Tensor<float> inputs = tz::tensor<float>({
        0.0f, 1.0f, 1.0f, 0.0f
    });
    outputs.reshape({ 4, 1, 1 });
    
    // Creating model
    size_t epochs = 10000;
    
    tz::Net<float> net;

    net.add_layer(LinearLayer({2, 3}), Tanh());
    net.add_layer(LinearLayer({3, 1}), Tanh());

    net.train(inputs, outputs, MSELoss(), 0.1, epochs);

    net.saveas("nets/xor_net.tzn");

    return 0;
}
```

### MNIST example

Mnist is a classical dataset used for ML. The implementation in Tezma is as follows:

```cpp

int main(int argc, char const *argv[])
{
    tz::Dataset dataset;
    dataset.from_csv("datasets/mnist/");

    auto&[inputs, outputs] = dataset.traning_data();

    // Creating model
    size_t epochs = 10000;
    
    tz::Net<float> net;

    net.add_layer(LinearLayer({32, 32}), Tanh());
    net.add_layer(LinearLayer({12, 12}), Tanh());
    net.add_layer(LinearLayer({10, 1}), Tanh());

    net.train(inputs, outputs, AdamOptimizer(), 0.1, epochs);

    net.saveas("nets/mnist_net.tzn");

    return 0;
}
```