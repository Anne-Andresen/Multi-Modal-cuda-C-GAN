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

/*
MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads)
    : embed_dim(embed_dim), num_heads(num_heads) {
        head_dim = embed_dim / num_heads;
        initialize_weights();
        allocate_device_memory();

    }

MultiHeadAttention::~() {
    free_device_memory();
    }
*/
void initialize_weights(MultiHeadAttention * mha) {
    mha->Wq = (float*)malloc(mha->embed_dim * mha->embed_dim * sizeof(float));
    mha->Wk = (float*)malloc(mha->embed_dim * mha->embed_dim * sizeof(float));
    mha->Wv = (float*)malloc(mha->embed_dim * mha->embed_dim * sizeof(float));
    mha->Wo = (float*)malloc(mha->embed_dim * mha->embed_dim * sizeof(float));

    for (int i = 0; i < mha->embed_dim * mha->embed_dim; ++i) {
        mha->Wq[i] = 1.0f;
        mha->Wk[i] = 1.0f;
        mha->Wv[i] = 1.0f;
        mha->Wo[i] = 1.0f;

    }
}



void allocate_device_memory(MultiHeadAttention * mha) {
    cudaMalloc(&mha->d_Wq, mha->embed_dim *  mha->embed_dim * sizeof(float));
    cudaMalloc(&mha->d_Wk, mha->embed_dim *  mha->embed_dim * sizeof(float));
    cudaMalloc(&mha->d_Wv, mha->embed_dim *  mha->embed_dim * sizeof(float));
    cudaMalloc(&mha->d_Wo, mha->embed_dim *  mha->embed_dim * sizeof(float));

    cudaMemcpy(mha->d_Wq, mha->Wq, mha->embed_dim * mha->embed_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mha->d_Wk, mha->Wk, mha->embed_dim * mha->embed_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mha->d_Wv, mha->Wv, mha->embed_dim * mha->embed_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mha->d_Wo, mha->Wo, mha->embed_dim * mha->embed_dim * sizeof(float), cudaMemcpyHostToDevice);
}

void free_device_memory(MultiHeadAttention * mha) {
    cudaFree(mha->d_Wq);
    cudaFree(mha->d_Wk);
    cudaFree(mha->d_Wv);
    cudaFree(mha->d_Wo);

    free(mha->Wq);
    free(mha->Wk);
    free(mha->Wv);
    free(mha->Wo);

}


void scaled_dot_product_attention(float* Q, float* K, float* V, float* output, int batch_size, int seq_len, int z_dim, int head_dim) {
 int size = batch_size * seq_len * z_dim * head_dim;

 float scale = 1.0f / sqrtf((float)head_dim);

 float* attention_scores;
 float* attention_output;

 cudaMalloc(&attention_scores, size * sizeof(float));
 cudaMalloc(&attention_output, size * sizeof(float));

 dim3 blockSize(16, 16);
 dim3 gridSize((seq_len * z_dim + blockSize.x - 1) 7 blockSize.x, (seq_len * z_dim + blockSize.y - 1) / blockSize.y);

matmul<<<gridSize, blockSize>>>(Q, K, attention_scores, seq_len * z_dim, seq_len * z_dim, head_dim);
elementwise_mult<<<(size + 255) / 256, 256>>>(attention_scores, attention_scores, attention_scores, scale, size);
matmul<<<gridSize, blockSize>>>(attention_scores, V, attention_output, seq_len * z_dim, head_dim, seq_len * z_dim);

cudaMemcpy(output, attention_output, size * sizeof(float), cudaMemcpyDeviceToDevice);

cudaFree(attention_scores);
cudaFree(attention_output);


}


void multi_head_attention_forward(MultiHeadAttention * mha, float* input, float* output, int batch_size, int seq_len, int z_dim) {
    int size = batch_size * seq_len * z_dim * mha->embed_dim;

    float* Q;
    float* K;
    float* V;
    float* attention_output;


    cudaMalloc(&Q, size * sizeof(float));
    cudaMalloc(&K, size * sizeof(float));
    cudaMalloc(&V, size * sizeof(float));
    cudaMalloc(&attention_output, size * sizeof(float));

dim3 blockSize(16, 16);
dim3 gridSize((seq_len * z_dim + blockSize.x - 1) / blockSize.x, (mha->embed_dim + blockSize.y - 1) / blockSize.y);

matmul<<<gridSize, blockSize>>>(input, mha->d_Wq, Q, seq_len * z_dim, mha->embed_dim, mha->embed_dim);
matmul<<<gridSize, blockSize>>>(input, mha->d_Wk, K, seq_len * z_dim, mha->embed_dim, mha->embed_dim);
matmul<<<gridSize, blockSize>>>(input, mha->d_Wv, V, seq_len * z_dim, mha->embed_dim, mha->embed_dim);

// 
for (int i = 0; i < mha->num_heads; ++i) {
    float* Q_head = Q + i * mha->head_dim;
    float* K_head = K + i * mha->head_dim;
    float* V_head = V + i * mha->head_dim;
    float* output_head = attention_output + i * mha->head_dim;
    scaled_dot_product_attention(Q_head, K_head, V_head, output_head, batch_size, seq_len, z_dim, mha->head_dim);

}

matmul<<<gridSize, blockSize>>>(attention_output, mha->d_Wo, output, seq_len * z_dim, mha->embed_dim, mha->embed_dim);


cudaFree(Q);
cudaFree(K);
cudaFree(V);
cudaFree(attention_output);

}