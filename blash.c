#include "blas.h"

void gemm(float alpha, int m, int n, int k, float beta, float* a, int lda, float* b, int ldb, float* c, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i * ldc + j] = 0;
            for (int l = 0; l < k; l++) {
                c[i * ldc + j] += alpha * a[i * lda + l] * b[l * ldb + j];
            }
            if (beta != 0) {
                c[i * ldc + j] *= beta;
            }
        }
    }
}
/*Redo to  arithmetic operations */