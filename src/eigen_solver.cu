#include "eigen_solver.h"
#include "kernels.h"

float normL2(float* d_partial, int num_blocks) {

    float* h_partial = (float*) malloc(num_blocks * sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(
        h_partial, d_partial,
        num_blocks * sizeof(float),
        cudaMemcpyDeviceToHost));

    float sum = 0.0f;

    for (int i = 0; i < num_blocks; i++) {
        sum += h_partial[i];
    }

    free(h_partial);

    return sqrtf(sum);
}


void power_iter(float* d_A, int n, float* lambda_out, float* v_out, int max_iter, float tol) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads;


    float *d_v, *d_Av, *d_partial;


    CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Av, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(float)));

    float *h_init = (float*) malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        h_init[i] = (float)rand() / RAND_MAX;
    }

    CUDA_CHECK(cudaMemcpy(d_v, h_init, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h_init);

    dot_prod<<<blocks, threads>>>(d_v, d_v, d_partial, n);
    float norm = normL2(d_partial, blocks);
    normalize<<<blocks, threads>>>(d_v, norm, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    float lambda_old = 0.0f;
    float lambda = 0.0f;


    for (int iter = 0; iter < max_iter; iter++) {

        mat_vec_mul<<<blocks, threads>>>(d_A, d_v, d_Av, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        dot_prod<<<blocks, threads>>>(d_v, d_Av, d_partial, n);
        lambda = normL2(d_partial, blocks);


        dot_prod<<<blocks, threads>>>(d_Av, d_Av, d_partial, n);
        norm = normL2(d_partial, blocks);

        normalize<<<blocks, threads>>>(d_Av, norm, n);

        copy_vec<<<blocks, threads>>>(d_Av, d_v, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (iter > 0 && fabsf(lambda - lambda_old) < tol) {
            printf("Converged in %d iterations\n", iter + 1);
            break;
        }

        lambda_old = lambda;

        if (iter == max_iter - 1) {
            printf("Reached max iterations\n");
        }
    }

    *lambda_out = lambda;

    CUDA_CHECK(cudaMemcpy(v_out, d_v, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_Av));
    CUDA_CHECK(cudaFree(d_partial));

    
}