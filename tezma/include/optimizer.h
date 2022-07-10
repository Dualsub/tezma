#pragma once

#include "tezma.h"
#include "tensor.h"

namespace tz
{
    template <typename Numeric>
    class Optimizer
    {
    protected:
        float m_learning_rate{0.0f};

    public:
        Optimizer() = default;
        Optimizer(float m_learning_rate) : m_learning_rate(m_learning_rate) {}
        ~Optimizer() = default;

        virtual void step(std::vector<Parameter<Numeric>> &params) = 0;
        void set_learning_rate(float learning_rate) { m_learning_rate = learning_rate; }
    };

    using DType = float;

    class SGDOptimizer : public Optimizer<DType>
    {
    public:
        SGDOptimizer() = default;
        SGDOptimizer(float m_learning_rate) : Optimizer(m_learning_rate) {}
        ~SGDOptimizer() = default;

        virtual void step(std::vector<Parameter<DType>> &params) override
        {
            TZ_ASSERT((params.size() > 0 && m_learning_rate > 0.0f) && "SGDOptimizer::step: invalid params");
            for (auto &param : params)
            {
                param.value() -= param.grad() * m_learning_rate;
            }
        }
    };

    class AdamOptimizer : public Optimizer<DType>
    {
    public:
        // Implement Adam optimizer step.
        // https://arxiv.org/pdf/1412.6980.pdf
        // https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
        virtual void step(std::vector<Parameter<DType>> &params) override
        {
            TZ_ASSERT(0 && "AdamOptimizer::step() is not implemented.");
        }
    };

};