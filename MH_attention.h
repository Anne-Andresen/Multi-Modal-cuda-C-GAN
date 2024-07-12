#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include <cuda_runtime.h>

class MultiHeadAttention {
public:
    MultiHeadAttention(int embed_dim, int num_heads);
    ~MultiHeadAttention();

    void forward(float* input, float* output, int batch_size, int seq_len);
private:
    int embed_dim;
    int num_heads;
    int head_dim;
    float *Ww, *Wk, *Wv, *Wo;
    float *d_Wq, *d_Wk, *d_Wv, *d_Wo;

    void initialize_weights();
    void allocate_device_memory();
    void free_device_memory();
    void scaled_dot_product_attention(float* Q, float* K, float* V, float* output, int batch_size, int seq_len);

};

#endif // MULTIHEAD_ATTENTION_H