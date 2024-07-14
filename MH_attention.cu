#include <MH_attention.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>


__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int dep = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int e = 0; e < K; ++e) {
            value += A[row * K + e] * B[e * N + col];
        }
        C[row * N + col] = value;
    }
}


__global__ void elementwise_mult(float* A, float* B, float* C, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = (A[idx] * B[idx]) * scale;
    }
}


MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads)
    : embed_dim(embed_dim), num_heads(num_heads) {
        head_dim = embed_dim / num_heads;
        initialize_weights();
        allocate_device_memory();

    }

MultiHeadAttention::~() {
    free_device_memory();
    }

void MultiHeadAttention::initialize_weights() {
    Wq = new float[embed_dim * embed_dim];
    Wk = new float[embed_dim * embed_dim];
    Wv = new float[embed_dim * embed_dim];
    Wo = new float[embed_dim * embed_dim];

    for (int i = 0; i < embed_dim * embed_dim; ++i) {
        Wq[i] = 1.0f;
        Wk[i] = 1.0f;
        Wv[i] = 1.0f;
        Wo[i] = 1.0f;

    }
}



