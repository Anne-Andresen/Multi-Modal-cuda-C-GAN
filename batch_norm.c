#include "batch_norm.h"
#include "blas.h"

BatchNorm* batch_norm_init(int size) {
    BatchNorm* bn = (BatchNorm*)malloc(sizeof(BatchNorm));
    bn->gamma = (float*)malloc(sizeof(float) * size);
    bn->beta = (float*)malloc(sizeof(float) * size);
    bn->running_mean = (float*)malloc(sizeof(float) * size);
    bn->running_var = (float*)malloc(sizeof(float) * size);
    bn->epsilon = 1e-5;
    bn->size = size;

    return bn;
}

void batch_norm_free(BatchNorm* bn) {
    free(gamma);
    free(beta);
    free(running_mean);
    free(running_var);
    free(bn);
}

void* batch_norm_forward(BatchNorm* bn, float* input) {
    int batch_size = input[0];
    int size = bn->size;

    float* mean = (float*)malloc(sizeof(float) * size);
    float* var = (float*)malloc(sizeof(float) * size);

    for (int i = 0; i < size; i++) {
        mean[i] = 0;
        for (int b = 0; b < batch_size; b++) {
            mean[i] += input[b * size + i];
        }
        mean[i] /= batch_size;

        var[i] = 0;
        for (int b = 0; b < batch_size; b++) {
            var[i] += (input[b * size + i] - mean[i]) * (input[b * size + i] - mean[i]);
        }
        var[i] /= batch_size;
        var[i] = sqrt(var[i] + bn->epsilon);
    }

    float* output = (float*)malloc(sizeof(float) * batch_size * size);
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < size; i++) {
            output[b * size + i] = (input[b * size + i] - mean[i]) / var[i] * bn->gamma[i] + bn->beta[i];
        }
    }

    free(mean);
    free(var);

    return output;
}