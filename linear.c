#include "linear.h"
#include "blash.h"

Linear* linear_init(int in_features, int out_features) {
    Linear* linear = (Linear*)malloc(sizeof(Linear));
    in_features = in_features;
    out_features = out_features;
    weights = (float*)malloc(sizeof(float) * out_features * in_channels);
    bias = (float*)malloc(sizeof(float) * out_features);
    
    return linear;
}
void linear_free(Linear* linear) {
    free(weights);
    free(bias);
    free(linear);
}

void linear_forward(Linear* linear, float* input) {
    int batch_size = input[0];
    int in_features = in_features;
    int out_features = out_features;

    float* output = (float*)malloc(sizeof(float) * batch_size * out_features);
    gemm(1, out_features, in_features, batch_size, 1, input, in_features, weights, out_features, 0, 0, out_channels, out_features);

    for (int b = 0; b < batch_size; b++){
        for (int i = 0; i < out_features; i++){
            output[b * out_features + i] += bias[i];
        }
    }
    return output;
}