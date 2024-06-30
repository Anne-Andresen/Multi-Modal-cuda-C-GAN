/*#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

typedef float tensor_type;

struct Tensor {
    tensor_type *data;
    int64_t size;
};

struct MultiheadAttention {
    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t query_desc;
    cudnnTensorDescriptor_t key_desc;
    cudnnTensorDescriptor_t value_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t weight_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnTensorDescriptor_t attn_output_desc;
    cudnnTensorDescriptor_t attn_scores_desc;
    cudnnTensorDescriptor_t attn_weights_desc;
    float *weight_data;
    float *bias_data;
    int64_t embed_dim;
    int64_t num_heads;
};

struct CrossAttention {
    MultiheadAttention attention;
};

void create_multihead_attention(MultiheadAttention *attention, int64_t embed_dim, int64_t num_heads) {
    cublasCreate(&attention->cublas_handle);
    cudnnCreate(&attention->cudnn_handle);

    attention->embed_dim = embed_dim;
    attention->num_heads = num_heads;

    int64_t head_dim = embed_dim / num_heads;

    cudnnCreateTensorDescriptor(&attention->query_desc);
    cudnnSetTensor4dDescriptor(attention->query_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, embed_dim, -1, 1, 1);

    cudnnCreateTensorDescriptor(&attention->key_desc);
    cudnnSetTensor4dDescriptor(attention->key_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, embed_dim, -1, 1, 1);

    cudnnCreateTensorDescriptor(&attention->value_desc);
    cudnnSetTensor4dDescriptor(attention->value_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, embed_dim, -1, 1, 1);

    cudnnCreateTensorDescriptor(&attention->output_desc);
    cudnnSetTensor4dDescriptor(attention->output_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, embed_dim, -1, 1, 1);

    cudnnCreateFilterDescriptor(&attention->weight_desc);
    cudnnSetFilter4dDescriptor(attention->weight_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, embed_dim, num_heads * head_dim, 1, 1);

    cudnnCreateConvolutionDescriptor(&attention->conv_desc);
    cudnnSetConvolution2dDescriptor(attention->conv_desc, 1, 1, embed_dim, embed_dim, 1, 1, 0, 0, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    cudnnCreateTensorDescriptor(&attention->bias_desc);
    cudnnSetTensor4dDescriptor(attention->bias_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, embed_dim, 1, 1, 1);

    cudnnCreateTensorDescriptor(&attention->attn_output_desc);
    cudnnSetTensor4dDescriptor(attention->attn_output_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, num_heads, -1, 1,


    */

#include <cross_attention.h>
#include <stdlib.h>
#include <stdio.h>


__global__ void linear_transform(const float* input, const float* weights, const float* output, int seq_length, int embed_dim, int head_dim) {
    int idx = blockIdx.x * blockIdx.x + threadIdx.x;
    if (idx < seq_length * head_dim) {
        int seq_idx = idx / head_dim;
        int head_idx = idx % head_dim;
        float sum = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            sum += input[seq_idx * embed_dim + i] * weights[i * head_dim + head_idx];
        }
        output[idx] = sum;
    }
}


__global__ void scaled_dot_product_attention(const float* Q, const float* K, const float* V, float* output, int seq_length, int head_dim) {

    int idx = blockIdx.x * blockIdx.x + blockIdx.x;

    if (idx < seq_length * head_dim) {
        int seq_idx = idx / head_dim;
        int head_idx = idx % head_dim;
        float sum = 0.0f;
        for (int i = 0; i < seq_length; i ++) {
            sum += Q[seq_idx * head_dim + head_idx] * K[i *head_dim + head_idx];
        }
        sum /= sqrtf(head_dim);
        output[idx] = sum * V[seq_idx * head_dim + head_idx];
    }
}

void cross_attention_init(CrossAttention* attn, int embed_dim, int num_heads, int batch_size, int D, int H, int W) {
    attn->embed_dim = embed_dim;
    attn->num_heads = num_heads;
    attn->head_dim = head_dim;

    size_t tensor_size = batch_size *  D * H * W * embed_dim * sizeof(float);
    size_t weight_size = embed_dim * attn->head_dim * sizeof(float);

    cudaMalloc(&attn->d_queries, tensor_size);
    cudaMalloc(&attn->d_keys, tensor_size);
    cudaMalloc(&attn->d_values, tensor_size);
    cudaMalloc(&attn->d_output, tensor_size);

    cudaMalloc(&attn->d_Wq, weight_size);
    cudaMalloc(&attn->d_Wk, weight_size);
    cudaMalloc(&attn->d_Wv, weight_size);

}

void cross_attention_set_input(CrossAttention* attn, const float* queries, const float* keys, const float* values) {
    size_t tensor_size = sizeof(float) * attn->embed_dim * attn->head_dim;
    cudaMemcpy(attn->d_queries, queries, tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(attn->d_keys, keys, tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(attn->d_values, values, tensor_size, cudaMemcpyHostToDevice);
}


void cross_attention_forward(CrossAttention* attn) {
    int seq_length = attn->embed_dim;
    int batch_size = 1;

    linear_transform<<<1, seq_length * attn->head_dim>>>(attn->d_queries, attn->d_Wq, attn->d_queries, seq_length, attn->embed_dim, attn->head_dim);
    linear_transform<<<1, seq_length * attn->head_dim>>>(attn->d_keys, attn->d_Wk, attn->d_keys, seq_length, attn->embed_dim, attn->head_dim);
    linear_transform<<<1, seq_length * attn->head_dim>>>(attn->d_values, attn->d_Wv, attn->d_values, seq_length, attn->embed_dim, attn->head_dim);
    
    scaled_dot_product_attention<<<1, seq_length * attn->head_dim>>>(attn->d_queries, attn->d_keys, attn->values, attn->d_output, seq_length, attn->head_dim);

    linear_transform<<<1, seq_length * attn->head_dim>>>(attn->d_output, attn->d_Wo, attn->d_output, seq_length, attn->embed_dim, attn->head_dim);

}


void cross_attention_get_output(CrossAttention* attn, float* output) {
    size_t tensor_size = sizeof(float) * attn->embed_dim * attn->head_dim;
    cudaMemcpy(output, attn->d_output, tensor_size, cudaMemcpyDeviceToHost);
}

void cross_attention_free(CrossAttention* attn) {
    cudaFree(attn->d_queries);
    cudaFree(attn->d_keys);
    cudaFree(attn->d_values);
    cudaFree(attn->d_output);
    cudaFree(attn->d_Wq);
    cudaFree(attn->d_Wk);
    cudaFree(attn->d_Wv);
    cudaFree(attn->d_Wo);
}