#pragma once

#include "tensor.h"
#include <iostream>
#include <fstream>

namespace tz
{
    enum class SerializationFormat
    {
        JSON,
        BINARY
    };

    enum class SerializationTypeCode : uint32_t
    {
        NULL_ = 0,
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        FLOAT32,
        FLOAT64,
        STRING,
        BOOL,
        ARRAY,
        OBJECT
    };

    class SerializationContext
    {
        // Stream to write to
        std::vector<char> m_data_buffer;
        size_t read_idx;

    public:
        SerializationContext() = default;
        ~SerializationContext() = default;

        void save(const std::string &filename)
        {
            auto fs = std::ofstream(filename, std::ios::binary);
            fs.write(m_data_buffer.data(), m_data_buffer.size());
            fs.close();
        }

        void write(const char *data, size_t size)
        {
            for (size_t i = 0; i < size; i++)
            {
                char data_byte = *(data + i * sizeof(char));
                m_data_buffer.push_back(data_byte);
            }
        }

        void open(const std::string &filename)
        {
            auto fs = std::ifstream(filename, std::ios::binary);
            m_data_buffer = std::vector<char>((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
            fs.close();
            read_idx = 0;
        }

        void read(char *data, size_t size)
        {
            for (size_t i = 0; i < size; i++)
            {
                char data_byte = m_data_buffer[read_idx + i];
                *(data + i * sizeof(char)) = data_byte;
            }
            read_idx += size;
        }
    };

    template <typename Numeric>
    Tensor<Numeric> load(const std::string &filename)
    {
        SerializationContext context;
        context.open(filename);
        return deserialize<Numeric>(context);
    }

    template <typename Numeric>
    void save(const Tensor<Numeric> &t, const std::string &filename)
    {
        SerializationContext context;
        serialize(t, context);
        context.save(filename);
    }

    template <typename Numeric>
    Tensor<Numeric> deserialize(SerializationContext &ctx)
    {
        TensorHeader header;

        ctx.read((char *)&header, sizeof(TensorHeader));

        std::vector<size_t> shape;
        for (size_t i = 0; i < header.order; i++)
        {
            uint32_t s;
            ctx.read((char *)&s, sizeof(uint32_t));
            shape.push_back((size_t)s);
        }

        Tensor<Numeric> t(shape);
        ctx.read(reinterpret_cast<char *>(t.data()), sizeof(Numeric) * t.size());

        return t;
    }

    template <typename Numeric>
    void serialize(const Tensor<Numeric> &t, SerializationContext &ctx)
    {
        TensorHeader header;

        header.size = (uint32_t)t.size();
        header.type_code = (uint32_t)SerializationTypeCode::FLOAT32;
        header.order = (uint32_t)t.order();

        ctx.write(reinterpret_cast<const char *>(&header), sizeof(TensorHeader));

        for (auto &&shape : t.shape())
        {
            uint32_t s = (uint32_t)shape;
            ctx.write((const char *)&s, sizeof(uint32_t));
        }

        ctx.write(reinterpret_cast<const char *>(t.data()), sizeof(Numeric) * t.size());
    }

    struct TensorHeader
    {
        uint32_t size;
        uint32_t type_code;
        uint32_t order;
    };

}