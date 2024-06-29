#ifndef CROSS_ATTENTION_H
#define CROSS_ATTENTION_H

#include <cuda_runtime.h>


typedef struct {
    int embed_dim;
    int num_heads;
    int head_dim;
    float *d_queries;
    float *d_keys;
    float *d_values;
    float *d_output;
    float *d_Wq;
    float *d_Wk;
    float *d_Wv;
    float *d_Wo;
    
    } CrossAttention;

void cross_attention_init(CrossAttention* attn, int embed_dim, int num_heads, int batch_size, int D, int H, int W);
void cross_attention_set_input(CrossAttention* attn, const float* queries, const float* keys, const float* values);
void cross_attention_forward(CrossAttention* attn);
void cross_attention_get_output(CrossAttention* attn, float* output);
void cross_attention_free(CrossAttention* attn);


#endif