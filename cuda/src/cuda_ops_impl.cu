#include "cuda_ops_impl.h"
#include <assert.h>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void gpu_matrix_mult(float *A, float *B, float *C, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < n && row < m)
    {
        for (int i = 0; i < k; i++)
        {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void cuda_matmul_impl(float *A, float *B, float *C, int m, int n, int k)
{
    dim3 dim_grid(ceilf(m/(float)BLOCK_SIZE), ceilf(n/(float)BLOCK_SIZE), 1);
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    gpu_matrix_mult<<<dim_grid, dim_block>>>(A, B, C, m, n, k);
    cudaDeviceSynchronize();
}

int CudaData::ALLOC_COUNT = 0;

CudaData::CudaData(size_t size)        
{
    m_size = size;
    int err = cudaMalloc((void **)&m_data, size * sizeof(float));
    assert(err == cudaSuccess && "cudaMalloc failed");
    // ALLOC_COUNT++;
    // std::cout << "CudaData alloc count: " << ALLOC_COUNT << std::endl;
}

CudaData::CudaData(const float *data, size_t size)        
{
    m_size = size;
    int err = cudaMalloc((void **)&m_data, size * sizeof(float));
    assert(err == cudaSuccess && "cudaMalloc failed");
    err = cudaMemcpy(m_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess && "cudaMemcpy failed");
    // ALLOC_COUNT++;
    // std::cout << "CudaData alloc count: " << ALLOC_COUNT << std::endl;
}

CudaData::~CudaData()
{
    cudaFree(m_data);
    // ALLOC_COUNT--;
    // std::cout << "CudaData alloc count: " << ALLOC_COUNT << std::endl;
}

float* CudaData::to_host()
{
    float *host_data = new float[m_size];
    int err = cudaMemcpy(host_data, m_data, m_size * sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess && "cudaMemcpy failed");
    return host_data;
}