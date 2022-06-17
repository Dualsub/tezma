#pragma once

#include <vector>
#include "tezma.h"
#include "layers.h"
#include "function.h"

namespace tz
{
    class Network
    {
    private:
        std::vector<Layer> m_layers;

    public:
        // A type alias. It is a way to define a new name for a type.
        using DType = float;

        Network();
        ~Network();

        void add_layer(const Layer &layer, const Function<DType> &activation_function);
    };
}
