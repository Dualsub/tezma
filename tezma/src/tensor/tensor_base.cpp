#include "tensor/tensor_base.h"

namespace tz
{

    std::default_random_engine rng_engine;

    void seed(unsigned int seed)
    {
        rng_engine.seed(seed);
    }

    std::ostream &operator<<(std::ostream &out, const tz::Shape &s)
    {
        if (!s.empty())
        {
            out << '(';
            std::copy(s.begin(), s.end()-1, std::ostream_iterator<size_t>(out, ", "));
            out << s.back();
            out << ')';
        }
        return out;
    }


}