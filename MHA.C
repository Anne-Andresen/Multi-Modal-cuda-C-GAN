include <stdio.h>
#include <cuda_runtime.h>
#include "multihead_attention.h"

int main() {
    // Define the dimensions and batch size
    int embed_dim = 128;
    int num_heads = 8;
    int batch_size = 32;
    int seq_len = 10;
    int z_dim = 10;

    // Instantiate the MultiHeadAttention structure
    MultiHeadAttention mha;
    MultiHeadAttention_init(&mha, embed_dim, num_heads);

    // Allocate and initialize input and output data
    float* input_data;
    float* output_data;
    cudaMalloc(&input_data, batch_size * seq_len * z_dim * embed_dim * sizeof(float));
    cudaMalloc(&output_data, batch_size * seq_len * z_dim * embed_dim * sizeof(float));

    // Initialize input_data with some values (example, for actual use case use real data)
    float* h_input_data = (float*)malloc(batch_size * seq_len * z_dim * embed_dim * sizeof(float));
    for (int i = 0; i < batch_size * seq_len * z_dim * embed_dim; ++i) {
        h_input_data[i] = 1.0f; // example value
    }
    cudaMemcpy(input_data, h_input_data, batch_size * seq_len * z_dim * embed_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Call the forward function
    MultiHeadAttention_forward(&mha, input_data, output_data, batch_size, seq_len, z_dim);

    // Use the output_data (example: print the first element)
    float* h_output_data = (float*)malloc(batch_size * seq_len * z_dim * embed_dim * sizeof(float));
    cudaMemcpy(h_output_data, output_data, batch_size * seq_len * z_dim * embed_dim * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Output[0]: %f\n", h_output_data[0]);

    // Free resources
    MultiHeadAttention_free(&mha);
    cudaFree(input_data);
    cudaFree(output_data);
    free(h_input_data);
    free(h_output_data);

    return 0;
}