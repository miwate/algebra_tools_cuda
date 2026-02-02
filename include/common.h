#ifndef COMMON_H
    #define COMMON_H

    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include <cuda_runtime.h>

    /* Check error */
    #define CUDA_CHECK(call) \
        do { \
            cudaError_t err = call; \
                if (err != cudaSuccess) { \
                fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                        cudaGetErrorString(err)); \
                exit(EXIT_FAILURE); \
            } \
        } while(0)

    /* Check kernel launch errors */
    #define CUDA_CHECK_KERNEL() \
        do { \
            cudaError_t err = cudaGetLastError(); \
            if (err != cudaSuccess) { \
                fprintf(stderr, "CUDA kernel error at %s:%d - %s\n", __FILE__, __LINE__, \
                        cudaGetErrorString(err)); \
                exit(EXIT_FAILURE); \
            } \
        } while(0)

#endif /* COMMON_H */