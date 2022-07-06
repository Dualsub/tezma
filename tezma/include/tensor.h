#pragma once

#include "tezma.h"
#include <initializer_list>
#include <vector>
#include <memory>
#include <iostream>
#include <random>
#include <assert.h>
#include <math.h>
#include <string>
#include <sstream>

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace tz
{

#pragma region Random

    extern std::default_random_engine rng_engine;
    void seed(unsigned int seed);

#pragma endregion

#pragma region Shape

    using Shape = std::vector<size_t>;

    std::ostream &operator<<(std::ostream &out, const tz::Shape &s);

#pragma endregion

#pragma region Class

    struct Range
    {
        size_t start;
        size_t end;

        Range(size_t start, size_t end)
            : start(start), end(end)
        {
            TZ_ASSERT(start <= end && "Invalid range.");
        }

        constexpr size_t size() const
        {
            return end - start;
        }
    };

    /* A class that represents a multi-dimensional array of numbers */
    template <typename Numeric>
    class Tensor
    {
    private:
        std::shared_ptr<Numeric> m_data;
        Shape m_shape;
        size_t m_size;

    public:
        Tensor() = default;
        ~Tensor() = default;
        Tensor<Numeric> &operator=(const Tensor<Numeric> &) = default;
        Tensor(const Tensor<Numeric> &) = default;
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
        Tensor(const Shape &shape)
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
        void reshape(const Shape &shape)
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
        const Shape &shape() const { return m_shape; }

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

        const Numeric *data() const { return m_data.get(); }

        /**
         * The function takes an output stream and a tensor as input, and returns the output stream
         * with the tensor's contents appended to it
         *
         * @return The ostream object.
         */
        friend std::ostream &operator<<(std::ostream &os, const Tensor<Numeric> &t)
        {
            if (t.order() == 2)
            {
                os << "[";
                for (size_t i = 0; i < t.shape(0); i++)
                {
                    os << "[";
                    size_t size = t.size();
                    if (size > 0)
                    {
                        os << t[0];
                        for (size_t i = 1; i < t.shape(1); i++)
                        {
                            os << ", " << t[i];
                        }
                    }
                    os << "],";
                    os << "\n";
                }
                os << "]";
            }
            else
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
            }

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

        Numeric &operator[](const std::initializer_list<size_t> idxs)
        {
            TZ_ASSERT(idxs.size() == m_shape.size() && "Indexes not matching with shape.");

            // Linear index into array.
            size_t idx = 0;

            // Loop count.
            size_t j = 0;

            // Size of the current block.
            size_t block_size = m_size;

            for (auto &i : idxs)
            {
                // The block size if one if the shape does not exist, so no divide in that case.
                if (j < m_shape.size())
                    block_size /= m_shape[j];

                idx += i * block_size;

                j++;
            }

            return m_data.get()[idx];
        }

        Tensor<Numeric> operator[](const Range range)
        {
            auto shape = m_shape;
            shape[0] = range.size();
            Tensor<Numeric> result(shape);
            auto block_size = result.size() / shape[0];
            // Copy the data.
            std::copy(m_data.get() + range.start * block_size, m_data.get() + range.end * block_size, result.data());

            return result;
        }

        Tensor<Numeric> slice(std::initializer_list<size_t> idxs) const
        {
            TZ_ASSERT(idxs.size() < m_shape.size() && "To many indexes.");

            Tensor<Numeric> result(Shape(m_shape.begin() + idxs.size(), m_shape.end()));

            // Linear index into array.
            size_t idx = 0;

            // Loop count.
            size_t j = 0;

            // Size of the current block.
            size_t block_size = m_size;

            for (auto &i : idxs)
            {
                // The block size if one if the shape does not exist, so no divide in that case.
                block_size /= m_shape[j];
                idx += i * block_size;
                j++;
            }

            // Will be calculated from constructor.
            size_t new_size = result.m_size;

            // Copy data.
            Numeric *dst = result.data();
            Numeric *src = m_data.get();
            memcpy_s(dst, new_size * sizeof(Numeric), src + idx, new_size * sizeof(Numeric));

            return result;
        }

        void set_slice(const std::initializer_list<size_t> idxs, Tensor &slice)
        {
            TZ_ASSERT(idxs.size() < m_shape.size() && "To many indexes.");

            // Linear index into array.
            size_t idx = 0;

            // Loop count.
            size_t j = 0;

            // Size of the current block.
            size_t block_size = m_size;

            for (auto &i : idxs)
            {
                // The block size if one if the shape does not exist, so no divide in that case.
                block_size /= m_shape[j];

                idx += i * block_size;

                j++;
            }

            // Will be calculated from constructor.
            size_t slice_size = slice.m_size;

            // Copy data.
            Numeric *dst = m_data.get();
            Numeric *src = slice.data();
            memcpy_s(dst + idx, slice_size * sizeof(Numeric), src, slice_size * sizeof(Numeric));
        }
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
    Tensor<Numeric> operator*(const Tensor<Numeric> &t, Numeric n)
    {
        Tensor<Numeric> result(t.shape());
        for (size_t i = 0; i < t.size(); i++)
            result[i] = t[i] * n;

        return result;
    }

    /* Defining the multiplication operator for the Tensor class. */
    template <typename Numeric>
    Tensor<Numeric> operator*(Numeric n, const Tensor<Numeric> &t) { return t * n; }
    template <typename Numeric>
    Tensor<Numeric> operator*(const Tensor<Numeric> &t, size_t n) { return t * (Numeric)n; }
    template <typename Numeric>
    Tensor<Numeric> operator*(size_t n, const Tensor<Numeric> &t) { return (Numeric)n * t; }

    /**
     * Subtract two tensors and return the result.
     *
     *
     * @param t1 The first tensor to subtract.
     * @param t2 The tensor to subtract to the current tensor.
     *
     * @return Resulting tensor.
     */
    template <typename Numeric>
    Tensor<Numeric> operator-(const Tensor<Numeric> &t1, const Tensor<Numeric> &t2)
    {
        return broadcast(t1, t2, [](Numeric a, Numeric b)
                         { return a - b; });
    }

    /**
     * Add two tensors together and return the result.
     *
     *
     * @param t1 The first tensor to add.
     * @param t2 The tensor to add to the current tensor.
     *
     * @return Resulting tensor.
     */
    template <typename Numeric>
    Tensor<Numeric> operator+(const Tensor<Numeric> &t1, const Tensor<Numeric> &t2)
    {
        return broadcast(t1, t2, [](Numeric a, Numeric b)
                         { return a + b; });
    }

    /**
     * Multiply two tensors together and return the result.
     *
     *
     * @param t1 The first tensor to multiply.
     * @param t2 The tensor to multiply to the current tensor.
     *
     * @return Resulting tensor.
     */
    template <typename Numeric>
    Tensor<Numeric> operator*(const Tensor<Numeric> &t1, const Tensor<Numeric> &t2)
    {
        return broadcast(t1, t2, [](Numeric a, Numeric b)
                         { return a * b; });
    }

    template <typename Numeric>
    void operator-=(Tensor<Numeric> &t1, const Tensor<Numeric> &t2)
    {
        t1 = t1 - t2;
    }

    template <typename Numeric>
    void operator+=(Tensor<Numeric> &t1, const Tensor<Numeric> &t2)
    {
        t1 = t1 + t2;
    }

    template <typename Numeric>
    void operator*=(Tensor<Numeric> &t1, const Tensor<Numeric> &t2)
    {
        t1 = t1 * t2;
    }

    template <typename Numeric>
    Tensor<Numeric> pow(Tensor<Numeric> &t, size_t n)
    {
        Tensor<Numeric> result(t.shape());
        for (size_t i = 0; i < t.size(); i++)
            result[i] = (Numeric)std::pow(t[i], n);

        return result;
    }

    template <typename Numeric>
    Tensor<bool> operator<(const Tensor<Numeric> &t, float f)
    {
        Tensor<bool> result(t.shape());
        for (size_t i = 0; i < t.size(); i++)
            result[i] = t[i] < f;

        return result;
    }

    template <typename Numeric>
    Tensor<bool> operator>(const Tensor<Numeric> &t, float f)
    {
        Tensor<bool> result(t.shape());
        for (size_t i = 0; i < t.size(); i++)
            result[i] = t[i] > f;

        return result;
    }

    template <typename Numeric>
    Tensor<bool> operator==(const Tensor<Numeric> &t, float f)
    {
        Tensor<bool> result(t.shape());
        for (size_t i = 0; i < t.size(); i++)
            result[i] = t[i] == f;

        return result;
    }

    template <typename Numeric>
    bool operator==(const Tensor<Numeric> &t1, const Tensor<Numeric> &t2)
    {
        if (t1.shape() != t2.shape())
            return false;

        for (size_t i = 0; i < t1.size(); i++)
            if (t1[i] == t2[i])
                ;

        return true;
    }

    template <typename Numeric>
    Tensor<Numeric> transpose(Tensor<Numeric> &t)
    {
        TZ_ASSERT(t.order() == 2 && "Can only transpose matrices.");

        Tensor<Numeric> result = zeros<Numeric>({t.shape(1), t.shape(0)});
        for (size_t i = 0; i < t.shape(0); i++)
        {
            for (size_t j = 0; j < t.shape(1); j++)
            {
                result[{j, i}] = t[{i, j}];
            }
        }
        return result;
    }

#pragma endregion

#pragma region Statistics

    /**
     * It takes a tensor and returns the mean of all the elements in the tensor
     *
     * @param t The tensor to calculate the mean of.
     *
     * @return The mean of the tensor.
     */
    template <typename Numeric>
    Numeric mean(const Tensor<Numeric> &t)
    {
        Numeric sum = 0;
        for (size_t i = 0; i < t.size(); i++)
        {
            sum += t[i];
        }

        return sum / (Numeric)t.size();
    }

    /**
     * > It calculates the standard deviation of a tensor
     *
     * @param t The tensor to calculate the standard deviation of.
     *
     * @return The standard deviation of the tensor.
     */
    template <typename Numeric>
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

    template <typename Numeric>
    Numeric max(const Tensor<Numeric> &t)
    {
        Numeric max = 0;
        for (size_t i = 0; i < t.size(); i++)
        {
            if (t[i] > max)
                max = t[i];
        }

        return max;
    }

    template <typename Numeric>
    Numeric min(const Tensor<Numeric> &t)
    {
        Numeric min = (std::numeric_limits<Numeric>::max())();
        for (size_t i = 0; i < t.size(); i++)
        {
            if (t[i] < min)
                min = t[i];
        }

        return min;
    }

    template <typename Numeric>
    size_t argmax(const Tensor<Numeric> &t)
    {
        Numeric max = 0;
        size_t index = 0;
        for (size_t i = 0; i < t.size(); i++)
        {
            if (t[i] > max)
            {
                max = t[i];
                index = i;
            }
        }

        return index;
    }

    template <typename Numeric>
    Numeric argmin(const Tensor<Numeric> &t)
    {
        Numeric min = (std::numeric_limits<Numeric>::min())();
        size_t index = 0;
        for (size_t i = 0; i < t.size(); i++)
        {
            if (t[i] < min)
            {
                min = t[i];
                index = i;
            }
        }

        return index;
    }

#pragma endregion

#pragma region Factories

    /**
     * It takes a list of sizes, and returns a tensor with those sizes, with empty memory.
     *
     * @param shape The shape of the tensor.
     *
     * @return A Tensor object.
     */
    template <typename Numeric>
    Tensor<Numeric> empty(std::initializer_list<size_t> shape)
    {
        Tensor<Numeric> result(shape);
        return result;
    }

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
     * It takes a list of sizes, and returns a tensor with those sizes, filled with zeros
     *
     * @param shape The shape of the tensor.
     *
     * @return A Tensor object.
     */
    template <typename Numeric>
    Tensor<Numeric> zeros(const Shape &shape)
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
    Tensor<Numeric> randu(std::initializer_list<size_t> shape)
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
            Shape new_shape;
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
        static size_t biggest = 0;
        size_t m = t1.shape(0);
        size_t k = t1.shape(1);
        size_t n = t1.shape(1);

        if(m+n+k > biggest)
        {
            biggest = m+n+k;
            std::cout << "Biggest: " << m << "x" << k << " * " << k << "x" << n << std::endl;
        }

        return matmul(t1, t2);
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
    Tensor<Numeric> matmul(const Tensor<Numeric> &t1, const Tensor<Numeric> &t2)
    {
        TZ_ASSERT((t1.order() == 2 && t2.order() == 2) && "Tensors are not matricies.");
        TZ_ASSERT((t1.shape(1) == t2.shape(0)) && "Matricies dimmensions are incompatible.");

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

    /**
     * > For each block of the result matrix, compute the dot product of the corresponding block of the
     * first matrix with the transpose of the corresponding block of the second matrix
     *
     * @param t1 The first matrix to multiply.
     * @param t2 The second matrix to multiply.
     *
     * @return A tensor of the same shape as the first argument.
     */
    template <typename Numeric>
    Tensor<Numeric> matmul_tiled(const Tensor<Numeric> t1, const Tensor<Numeric> &t2, const int block_size)
    {
        TZ_ASSERT((t1.order() == 2 && t2.order() == 2) && "Tensors are not matricies.");
        TZ_ASSERT((t1.shape(1) == t1.shape(0)) && "Matricies dimmensions are incompatible.");

        size_t rows = t1.shape(0);
        size_t cols = t2.shape(1);
        size_t m = t1.shape(1);

        Tensor<Numeric> result = zeros<Numeric>({rows, cols});
        for (size_t bi = 0; bi < rows; bi += block_size)
        {
            for (size_t bj = 0; bj < cols; bj += block_size)
            {
                for (size_t i = bi; i < bi + block_size; i++)
                {
                    for (size_t j = bj; j < bj + block_size; j++)
                    {
                        Numeric acc = 0;
                        for (size_t k = 0; k < m; k++)
                        {
                            acc += t1[i * m + k] * t2[k * cols + j];
                        }

                        result[i * cols + j] = acc;
                    }
                }
            }
        }

        return result;
    }

#pragma endregion

#pragma region Broadcasting

    /**
     * It takes two tensors and a binary function, and returns a tensor that is the result of applying the
     * function to each element of the two tensors
     *
     * @param a The first tensor.
     * @param b The second tensor.
     * @param func The function to be applied to the tensors.
     *
     * @return A tensor with the result of the binary operation.
     */
    template <typename Numeric, typename Func>
    Tensor<Numeric> broadcast(const Tensor<Numeric> &a, const Tensor<Numeric> &b, Func &&func)
    {
        // Compute the shape of the result tensor.

        /* Calculating the shape of the output tensor. */
        size_t block_size = 1;
        Shape out_shape;

        size_t P = std::max(a.order(), b.order());
        for (size_t i = 0; i < P; i++)
        {
            size_t a_shape = a.order() > i ? a.shape(a.order() - i - 1) : 1;
            size_t b_shape = b.order() > i ? b.shape(b.order() - i - 1) : 1;

            out_shape.insert(out_shape.begin(), std::max(a_shape, b_shape));

            if (a_shape == b_shape)
                block_size *= a_shape;
        }

        size_t num_blocks = std::max(a.size(), b.size()) / block_size;

        bool is_a_bigger = a.size() > b.size();

        /* Performing a binary operation on two tensors. */
        Tensor<Numeric> result(out_shape);
        for (size_t block = 0; block < num_blocks; block++)
        {
            size_t block_offset = block * block_size;
            for (size_t i = 0; i < block_size; i++)
            {
                size_t x = is_a_bigger ? block_offset + i : i;
                size_t y = !is_a_bigger ? block_offset + i : i;
                result[block_offset + i] = func(a[x], b[y]);
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
        TZ_ASSERT(0 && "Not implemented.");
        return {0, nullptr};
    }

#pragma endregion
}
