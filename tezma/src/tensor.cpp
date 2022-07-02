#include "tensor.h"

namespace tz {

    std::default_random_engine rng_engine;

    void seed(unsigned int seed)
    {
        rng_engine.seed(seed);
    }

}