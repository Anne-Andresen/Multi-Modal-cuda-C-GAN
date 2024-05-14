#include <stdio.h>
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
