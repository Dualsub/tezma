#pragma once

#include <initializer_list>
#include <vector>
#include <memory>
#include <iostream>
#include <random>
#include <assert.h>
#include <math.h>
#include <string>
#include <sstream>
#include "tezma.h"

namespace tz
{

#pragma region State

    static std::default_random_engine rng_engine;

    void seed(unsigned int seed)
    {
        rng_engine.seed(seed);
    }

#pragma endregion

#pragma region Class

    /* A class that represents a multi-dimensional array of numbers */
    template <typename Numeric>
    class Tensor
    {
    private:
        std::shared_ptr<Numeric> m_data;
        std::vector<size_t> m_shape;
        size_t m_size;

    public:
        Tensor() = default;
        ~Tensor() = default;

        /**
         * This function takes a list of numbers and creates a tensor with the shape of the list
         *
         * @param values The initializer list of values to be stored in the tensor.
         */
        Tensor(std::initializer_list<Numeric> values)
        {
            size_t size = values.size();
            m_size = size;
            m_shape = {size};

            m_data = std::shared_ptr<Numeric>((Numeric *)malloc(sizeof(Numeric) * size));

            size_t i = 0;
            for (auto &&value : values)
            {
                m_data.get()[i] = value;
                i++;
            }
        }

        /**
         * It takes a list of lists of numbers and creates a tensor from it
         *
         * @param initializer_list This is a C++11 feature that allows you to initialize a list of
         * values.
         */
        Tensor(std::initializer_list<std::initializer_list<Numeric>> values)
        {
            size_t rows = values.size();
            size_t cols = (*values.begin()).size();
            m_shape = {rows, cols};

            size_t size = 1;
            for (auto s : m_shape)
                size *= s;
            m_size = size;
            m_data = std::shared_ptr<Numeric>((Numeric *)malloc(sizeof(Numeric) * size));

            size_t i = 0;
            for (auto &&columns : values)
            {
                size_t j = 0;
                for (auto &&value : columns)
                {
                    m_data.get()[i * cols + j] = value;
                    j++;
                }
                i++;
            }
        }

        /**
         * This function takes a list of integers and creates a tensor with the specified shape
         *
         * @param shape The shape of the tensor.
         */
        Tensor(std::initializer_list<size_t> shape)
        {
            m_shape = shape;

            size_t size = 1;
            for (auto s : shape)
                size *= s;
            m_size = size;

            m_data = std::shared_ptr<Numeric>((Numeric *)malloc(sizeof(Numeric) * size));
        }

        /**
         * The function takes a vector of size_t's and creates a Tensor object with the shape of the
         * vector
         *
         * @param shape The shape of the tensor.
         */
        Tensor(const std::vector<size_t> &shape)
        {
            m_shape = shape;

            size_t size = 1;
            for (auto s : shape)
                size *= s;
            m_size = size;

            m_data = std::shared_ptr<Numeric>((Numeric *)malloc(sizeof(Numeric) * size));
        }

        /**
         * > Reshape the tensor to the new shape
         *
         * @param shape The new shape of the tensor.
         */
        void reshape(const std::initializer_list<size_t> shape)
        {
            size_t size = 1;
            for (auto s : shape)
                size *= s;
            TZ_ASSERT(size == m_size && "The new shape must have equal number of elements to the last.");

            m_shape = shape;
        }

        /**
         * > Reshape the tensor to the new shape
         *
         * @param shape The new shape of the tensor.
         */
        void reshape(const std::vector<size_t> &shape)
        {
            size_t size = 1;
            for (auto s : shape)
                size *= s;
            TZ_ASSERT(size == m_size && "The new shape must have equal number of elements to the last.");

            m_shape = shape;
        }

        /**
         * > If the size of the numeric type is different, then copy the data from the old tensor to
         * the new tensor
         *
         * @return A new tensor with the same shape as the original tensor, but with a different
         * numeric type.
         */
        template <typename NewNumeric>
        Tensor<NewNumeric> astype()
        {
            Tensor<NewNumeric> result(m_shape);
            if (sizeof(Numeric) != sizeof(NewNumeric))
            {
                std::copy(m_data.get(), m_data.get() + m_size, result.getData());
            }

            return result;
        }

        /**
         * It returns a constant reference to the shape of the tensor
         *
         * @return The shape of the tensor.
         */
        const std::vector<size_t> &shape() const { return m_shape; }

        /**
         * `shape` returns the size of the `i`th dimension of the tensor
         *
         * @param i the index of the dimension to get the size of
         *
         * @return The shape of the array.
         */
        size_t shape(size_t i) const { return m_shape[i]; }

        /**
         * `order()` returns the number of dimensions of the array
         *
         * @return The number of dimensions of the array.
         */
        size_t order() const { return m_shape.size(); }

        /**
         * It returns the size of the array.
         *
         * @return The size of the array.
         */
        size_t size() const { return m_size; }

        /**
         * Return a pointer to the data array.
         *
         * @return A pointer to the data.
         */
        Numeric *data() { return m_data.get(); }

        /**
         * Return a const pointer to the data array.
         *
         * @return A pointer to the data.
         */
        const Numeric *data() const { return m_data.get(); }

        /**
         * The function takes an output stream and a tensor as input, and returns the output stream
         * with the tensor's contents appended to it
         *
         * @return The ostream object.
         */
        friend std::ostream &operator<<(std::ostream &os, const Tensor<Numeric> &t)
        {
            os << "[";

            size_t size = t.size();
            if (size > 0)
            {
                os << t[0];
                for (size_t i = 1; i < size; i++)
                {
                    os << ", " << t[i];
                }
            }
            os << "]";

            return os;
        }

        /**
         * Returns the value of the element at the specified index.
         *
         * @param i The index of the element to access.
         *
         * @return A reference to the value at the given index.
         */
        Numeric operator[](size_t i) const { return m_data.get()[i]; }
        /**
         * Returns the value of the element at the specified index.
         *
         * @param i The index of the element to access.
         *
         * @return A reference to the value at the given index.
         */
        Numeric &operator[](size_t i) { return m_data.get()[i]; }
    };

#pragma endregion

#pragma region Operators

    /**
     * It multiplies a tensor by a scalar
     *
     * @param t The tensor to multiply.
     * @param n The number to multiply the tensor by.
     *
     * @return A new tensor with the same shape as the original tensor, but with each element
     * multiplied by the scalar.
     */
    template <typename Numeric>
    Tensor<Numeric> operator*(Tensor<Numeric> &t, Numeric n)
    {
        Tensor<Numeric> result(t.shape());
        for (size_t i = 0; i < t.size(); i++)
            result[i] = t[i] * n;

        return result;
    }

    /* Defining the multiplication operator for the Tensor class. */
    template <typename Numeric>
    Tensor<Numeric> operator*(Numeric n, Tensor<Numeric> &t) { return t * n; }
    template <typename Numeric>
    Tensor<Numeric> operator*(Tensor<Numeric> &t, size_t n) { return t * (Numeric)n; }
    template <typename Numeric>
    Tensor<Numeric> operator*(size_t n, Tensor<Numeric> &t) { return (Numeric)n * t; }

    /**
     * "Add two tensors together and return the result."
     *
     *
     * @param t1 The first tensor to add.
     * @param t2 The tensor to add to the current tensor.
     *
     * @return A new tensor with the same shape as the two input tensors.
     */
    template <typename Numeric>
    Tensor<Numeric> operator+(const Tensor<Numeric> &t1, const Tensor<Numeric> &t2)
    {
        TZ_ASSERT(t1.shape() == t2.shape() && "Tensors do not have the same shape.");

        Tensor<Numeric> result(t1.shape());
        for (size_t i = 0; i < t1.size(); i++)
            result[i] = t1[i] + t2[i];

        return result;
    }

#pragma endregion

#pragma region Statistics

    template <typename Numeric>
    /**
     * It takes a tensor and returns the mean of all the elements in the tensor
     *
     * @param t The tensor to calculate the mean of.
     *
     * @return The mean of the tensor.
     */
    Numeric mean(const Tensor<Numeric> &t)
    {
        Numeric sum = 0;
        for (size_t i = 0; i < t.size(); i++)
        {
            sum += t[i];
        }

        return sum / (Numeric)t.size();
    }

    template <typename Numeric>
    /**
     * > It calculates the standard deviation of a tensor
     *
     * @param t The tensor to calculate the standard deviation of.
     *
     * @return The standard deviation of the tensor.
     */
    Numeric std(const Tensor<Numeric> &t)
    {
        Numeric mu = mean(t);
        Numeric sum = 0;
        for (size_t i = 0; i < t.size(); i++)
        {
            Numeric inner = (t[i] - mu);
            sum += inner * inner;
        }

        return sqrt(sum / (Numeric)t.size());
    }

#pragma endregion

#pragma region Factories

    /**
     * It creates a tensor with the given valuers
     *
     * @param values The values of the tensor.
     *
     * @return A tensor filled with the values.
     */
    template <typename Numeric>
    Tensor<Numeric> tensor(std::initializer_list<std::initializer_list<Numeric>> values)
    {
        return Tensor<Numeric>(values);
    }

    /**
     * It creates a tensor with the given valuers
     *
     * @param values The values of the tensor.
     *
     * @return A tensor filled with the values.
     */
    template <typename Numeric>
    Tensor<Numeric> tensor(std::initializer_list<Numeric> values)
    {
        return Tensor<Numeric>(values);
    }

    /**
     * It creates a tensor with the given shape and fills it with ones
     *
     * @param shape The shape of the tensor.
     *
     * @return A tensor with all elements set to 1.
     */
    template <typename Numeric>
    Tensor<Numeric> ones(std::initializer_list<size_t> shape)
    {
        Tensor<Numeric> result(shape);
        for (size_t i = 0; i < result.size(); i++)
        {
            result[i] = 1;
        }

        return result;
    }

    /**
     * It takes a list of sizes, and returns a tensor with those sizes, filled with zeros
     *
     * @param shape The shape of the tensor.
     *
     * @return A Tensor object.
     */
    template <typename Numeric>
    Tensor<Numeric> zeros(std::initializer_list<size_t> shape)
    {
        Tensor<Numeric> result(shape);
        memset(result.data(), 0, result.size() * sizeof(Numeric));
        return result;
    }

    /**
     * It creates a tensor with the given shape and fills it with random numbers
     *
     * @param shape The shape of the tensor.
     *
     * @return A Tensor object.
     */
    template <typename Numeric>
    Tensor<Numeric> rand(std::initializer_list<size_t> shape)
    {
        Tensor<Numeric> result(shape);
        for (size_t i = 0; i < result.size(); i++)
        {
            result[i] = (Numeric)rand() / RAND_MAX;
        }

        return result;
    }

    /**
     * It creates a tensor of the given shape, and fills it with random numbers drawn from a normal
     * distribution with mean 0 and standard deviation 1
     *
     * @param shape The shape of the tensor.
     *
     * @return A tensor of random numbers.
     */
    template <typename Numeric>
    Tensor<Numeric> randn(std::initializer_list<size_t> shape)
    {
        std::normal_distribution<Numeric> d{0, 1};
        Tensor<Numeric> result(shape);
        for (size_t i = 0; i < result.size(); i++)
        {
            result[i] = d(rng_engine);
        }

        return result;
    }

#pragma endregion

#pragma region Operations

    /**
     * It removes the dimension of size 1 from the tensor
     *
     * @param t The tensor to be squeezed.
     * @param axis The axis to be squeezed.
     *
     * @return A new tensor with the same data as the input tensor, but with the specified axis
     * removed.
     */
    template <typename Numeric>
    Tensor<Numeric> squeeze(const Tensor<Numeric> &t, size_t axis = -1)
    {
        Tensor<Numeric> result = t;
        if (axis == -1)
        {
            result.reshape({t.size()});
        }
        else
        {
            TZ_ASSERT(t.shape(axis) == 1 && "Axis cannot be squeezed unless it is not 1.");
            std::vector<size_t> new_shape;
            for (size_t i = 0; i < t.order(); i++)
            {
                if (i != axis)
                    new_shape.push_back(t.shape(i));
            }

            result.reshape(new_shape);
        }

        return result;
    }

    template <typename Numeric>
    Tensor<Numeric> dot(const Tensor<Numeric> &t1, const Tensor<Numeric> &t2)
    {
    }

    /**
     * > We loop over the rows of the first matrix, the columns of the second matrix, and the columns
     * of the first matrix (which are the rows of the second matrix)
     *
     * @param t1 The first tensor.
     * @param t2 The second tensor to multiply.
     *
     * @return A Tensor<Numeric>
     */
    template <typename Numeric>
    Tensor<Numeric> matmul(Tensor<Numeric> &t1, Tensor<Numeric> &t2)
    {
        TZ_ASSERT((t1.order() == 2 && t2.order() == 2) && "Tensors are not matricies.");
        TZ_ASSERT((t1.shape(1) == t1.shape(0)) && "Matricies dimmensions are incompatible.");

        size_t rows = t1.shape(0);
        size_t cols = t2.shape(1);
        size_t m = t1.shape(1);

        Tensor<Numeric> result = zeros<Numeric>({rows, cols});
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < cols; j++)
            {
                for (size_t k = 0; k < m; k++)
                {
                    result[i * cols + j] += t1[i * m + k] * t2[k * cols + j];
                }
            }
        }

        return result;
    }

    template <typename Numeric>
    /**
     * > For each block of the result matrix, compute the dot product of the corresponding block of the
     * first matrix with the transpose of the corresponding block of the second matrix
     *
     * @param t1 The first matrix to multiply.
     * @param t2 The second matrix to multiply.
     *
     * @return A tensor of the same shape as the first argument.
     */
    Tensor<Numeric> matmul_tiled(Tensor<Numeric> &t1, Tensor<Numeric> &t2)
    {
        TZ_ASSERT((t1.order() == 2 && t2.order() == 2) && "Tensors are not matricies.");
        TZ_ASSERT((t1.shape(1) == t1.shape(0)) && "Matricies dimmensions are incompatible.");

        const int block_size = 64 / sizeof(Numeric);
        size_t N = t1.shape(0);
        size_t M = t2.shape(1);
        size_t K = t1.shape(1);

        Tensor<Numeric> result = zeros<Numeric>({M, N});
        for (size_t i0 = 0; i0 < N; i0 += block_size)
        {
            size_t imax = i0 + block_size > N ? N : i0 + block_size;

            for (size_t j0 = 0; j0 < M; j0 += block_size)
            {
                size_t jmax = j0 + block_size > M ? M : j0 + block_size;

                for (size_t k0 = 0; k0 < K; k0 += block_size)
                {
                    size_t kmax = k0 + block_size > K ? K : k0 + block_size;

                    for (size_t j1 = j0; j1 < jmax; ++j1)
                    {
                        size_t sj = M * j1;

                        for (size_t i1 = i0; i1 < imax; ++i1)
                        {
                            size_t mi = M * i1;
                            size_t ki = K * i1;
                            size_t kij = ki + j1;

                            for (size_t k1 = k0; k1 < kmax; ++k1)
                            {
                                result[kij] += t1[mi + k1] * t2[sj + k1];
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

#pragma endregion

#pragma region Serialization

    /**
     * It takes a tensor and returns a string that contains the tensor's shape and data in JSON format
     *
     * @param t The tensor to serialize
     * @param indent The number of spaces to indent the JSON.
     *
     * @return A string
     */
    template <typename Numeric>
    std::string to_json(const Tensor<Numeric> &t, size_t indent = 1)
    {
        std::ostringstream stream;
        stream << "{";

        stream << "\"shape\": [" << t.shape(0);

        for (size_t i = 1; i < t.order(); i++)
        {
            stream << "," << t.shape(i);
        }

        stream << "],";

        stream << "\"data\": [" << t[0];

        for (size_t i = 1; i < t.size(); i++)
        {
            stream << "," << t[i];
        }

        stream << "]";

        stream << "}";

        return stream.str();
    }

    template <typename Numeric>
    std::pair<size_t, char *> to_bytes(const Tensor<Numeric> &t)
    {
        TZ_ASSERT(0 && "Not implemened.");
        return {0, nullptr};
    }

#pragma endregion
}
