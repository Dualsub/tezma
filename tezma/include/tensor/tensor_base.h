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

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace tz
{

enum class DeviceType
{
    TZ_DEVICE_CPU,
    TZ_DEVICE_GPU_CUDA
};

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
        DeviceType m_device{DeviceType::TZ_DEVICE_CPU};

    public:
        Tensor() = default;
        ~Tensor() = default;

        Tensor(Numeric *data, std::initializer_list<size_t> shape)
        {
            m_shape = shape;

            size_t size = 1;
            for (auto s : shape)
                size *= s;
            m_size = size;

            m_data = std::shared_ptr<Numeric>(data);
        }

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

        // Sets the device type of the tensor.
        void device(DeviceType device)
        {
            auto old_device = m_device;
            m_device = device;

            if (old_device != device)
            {
                // GPU -> CPU
                if (device == DeviceType::TZ_DEVICE_CPU && old_device == DeviceType::TZ_DEVICE_GPU)
                {
                    return;
                }
                // CPU -> GPU
                else if (device == DeviceType::TZ_DEVICE_GPU && old_device == DeviceType::TZ_DEVICE_CPU)
                {
                    return;
                }
            }
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
}
