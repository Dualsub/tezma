#pragma once

#include "function.h"
#include "optimizer.h"

namespace tz
{
    #define TZ_LAYER(name) \
    virtual const std::string& type() override { static std::string s = #name; return s; }

    using DType = float;

    class Layer : public Function<DType>
    {
    public:
    private:
        std::vector<Parameter<DType>> m_params;
    public:
        virtual Tensor<DType> forward(const Tensor<DType> &t) override
        {
            return t;
        }

        virtual Tensor<DType> backward(const Tensor<DType> &t) override
        {
            return t;
        }
    };

    class LinearLayer : public Function<DType>
    {
    public:
        TZ_LAYER(LinearLayer)

        LinearLayer(size_t input_size, size_t output_size)
        {
            m_weights = tz::randn<DType>({input_size, output_size});
            m_bias = tz::randn<DType>({1, output_size});
        }

    private:
        float m_learning_rate = 0.01f;
        Tensor<DType> m_input;
        Tensor<DType> m_weights;
        Tensor<DType> m_bias;
        Tensor<DType> m_weights_grad;
        Tensor<DType> m_bias_grad;
    public:

        virtual Tensor<DType> forward(const Tensor<DType> &t) override
        {
            m_input = t;
            return dot(t, m_weights) + m_bias;
        }

        virtual Tensor<DType> backward(const Tensor<DType> &t) override
        {
            auto weights_gradient = dot(transpose(m_input), t); 
            auto input_gradient = dot(t, transpose(m_weights));
            
            m_weights_grad = weights_gradient;
            m_bias_grad = t;

            // m_weights = m_weights - (m_learning_rate * weights_gradient);
            // m_bias = m_bias - (m_learning_rate * t);

            return input_gradient;
        }

        const Tensor<DType>& weights() const
        {
            return m_weights;
        }

        const Tensor<DType>& bias() const
        {
            return m_bias;
        }

        std::vector<Parameter<DType>> params()
        {
            return {
                Parameter<DType>(m_weights, m_weights_grad),
                Parameter<DType>(m_bias, m_bias_grad)
            };
        }
    };


    class Conv2D : public Function<DType>
    {
    private:
        Shape m_kernal_shape;
        Shape m_output_shape;
        Shape m_input_shape;
        size_t m_depth;
        size_t m_input_depth;
        
        Tensor<DType> m_input;
        Tensor<DType> m_kernals;
        Tensor<DType> m_kernals_grad;
        Tensor<DType> m_biases;
        Tensor<DType> m_biases_grad;
        
    public:
        TZ_LAYER(Conv2D)

        Conv2D(const Shape& input_shape, size_t kernal_size, size_t depth) 
            : m_depth(depth),  m_input_shape(input_shape)
        {
            m_input_depth = input_shape[0];
            m_output_shape = { depth, input_shape[1] - kernal_size + 1, input_shape[2] - kernal_size + 1 };
            m_kernal_shape = { depth, input_shape[0], kernal_size, kernal_size };
            
            m_kernals = randn<DType>(m_kernal_shape);
            m_biases = randn<DType>(m_output_shape);
        }
    private:

    public:

        virtual Tensor<DType> forward(const Tensor<DType> &t) override
        {
            m_input = t;
            Tensor<DType> result = m_biases;

            // for (size_t i = 0; i < m_depth; i++)
            // {
            //     for (size_t j = 0; j < m_input_depth; j++)
            //     {
            //         result += correlate_valid(t.slice({ i }), m_kernals.slice({ i, j }));
            //     }
            // }

            // result = m_func->forward(result);
            return result;
        }

        virtual Tensor<DType> backward(const Tensor<DType> &t) override
        {
            Tensor<DType> kernals_gradient(m_kernal_shape);
            Tensor<DType> input_gradient(m_input_shape);

            // for (size_t i = 0; i < m_depth; i++)
            // {
            //     for (size_t j = 0; j < m_input_depth; j++)
            //     {
            //         kernals_gradient.set_slice({i,j}, correlate_valid(m_input.slice({ j }), grad.slice({ i }) ));
            //         input_gradient.add_slice({ j }, convolute_full(grad.slice({ i }), m_kernals.slice({i,j}) ));
            //     }
            // }

            // m_kernals_grad = kernals_gradient;
            // m_biases_grad = t;
            
            return input_gradient;
        }

        std::vector<Parameter<DType>> params()
        {
            return {
                Parameter(m_kernals, m_kernals_grad),
                Parameter(m_biases, m_biases_grad)
            };
        }

    };
}