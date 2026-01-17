#ifndef KERNELS_H

    #define KERNELS_H

    #include "common.h"

    __global__ void mat_vec_mul(const float* A, const float* v, float* result, int n);

    __global__ void dot_prod(const float* v, const float* w, float* result, int n);

    __global__ void normalize(float* v, float norm, int n);
    
    __global__ void copy_vec(const float* src, float* dst, int n);

#endif /* KERNELS_H */