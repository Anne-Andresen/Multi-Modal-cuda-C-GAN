#include "cross_attention.h"
#include <stdio.h>
#include <stdlib.h>

#define EMBED_DIM 64
#define NUM_HEADS 8
#define BATCH_SIZE 1
#define DEPTH 32
#define HEIGHT 32
#define WIDTH 32

int main() {
    size_t tensor_size = BATCH_SIZE * DEPTH * HEIGHT * WIDTH * EMBED_DIM * sizeof(float);
    float *queries = (float*)malloc(tensor_size);
    float *keys = (float*)malloc(tensor_size);
    float *values = (float*)malloc(tensor_size);
    float *output = (float*)malloc(tensor_size);

    // Initialize queries, keys, and values here
    for (int i = 0; i < BATCH_SIZE * DEPTH * HEIGHT * WIDTH * EMBED_DIM; ++i) {
        queries[i] = 1.0f;
        keys[i] = 1.0f;
        values[i] = 1.0f;
    }

    CrossAttention attn;
    cross_attention_init(&attn, EMBED_DIM, NUM_HEADS, BATCH_SIZE, DEPTH, HEIGHT, WIDTH);

    cross_attention_set_input(&attn, queries, keys, values);
    cross_attention_forward(&attn);
    cross_attention_get_output(&attn, output);

    // Print the output for verification
    printf("Output:\n");
    for (int i = 0; i < BATCH_SIZE * DEPTH * HEIGHT * WIDTH * EMBED_DIM; ++i) {
        printf("%f ", output[i]);
        if ((i + 1) % WIDTH == 0) printf("\n");
        if ((i + 1) % (WIDTH * HEIGHT) == 0) printf("\n");
    }

    cross_attention_free(&attn);

    free(queries);
    free(keys);
    free(values);
    free(output);

    return 0;
}
