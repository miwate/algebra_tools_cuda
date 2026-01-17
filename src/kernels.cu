#include "kernels.h"


__global__ void mat_vec_mul(const float* A, const float* v, float* result, int n) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float sum = 0.0f;

        for (int j = 0; j < n; j++) {
            sum += A[i*n+j] * v[j];
        }

        result[i] = sum;
    }
}


__global__ void dot_prod(const float* v, const float* w, float* result, int n) {

    __shared__ float cache[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = 0.0f;

    if (i < n) {
        temp = v[i] * w[i];
    }

    cache[tid] = temp;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1 ) {

        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = cache[0];
    }
}

__global__ void normalize(float* v, float norm, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        v[i] /= norm;
    }
}


__global__ void copy_vec(const float* src, float* dst, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        dst[i] = src[i];
    }
}