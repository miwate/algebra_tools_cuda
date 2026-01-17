#ifndef SOLVER_H

    #define SOLVER_H

    float normL2(float* d_partial, int num_blocks);

    void power_iter(float* d_A, int n, float* lambda_out, float* v_out, int max_iter, float tol);

#endif /* SOLVER_H */