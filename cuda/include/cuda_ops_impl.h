#pragma once

void cuda_matmul_impl(float *A, float *B, float *C, int m, int n, int k);

class CudaData
{
    size_t m_size;
    float *m_data;

public:
    static int ALLOC_COUNT;

    CudaData(size_t size);
    CudaData(const float *data, size_t size);
    ~CudaData();

    float *to_host();
    float *data() { return m_data; }
    size_t size() { return m_size; }
};
