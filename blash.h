#ifndef BLAS_H
#define BLAS_H

void gemm(float alpha, int m, int n, int k, float beta, float* a, int lda, float* b, int ldb, float* c, int ldc);

#endif  // BLAS_H