#include "tezma.h"
#include "tensor.h"
#include <iostream>

int main(int argc, char const *argv[])
{
    tz::Tensor<float> t = tz::randn<float>({32, 32, 1});
    std::cout << t << std::endl;
    return 0;
}
