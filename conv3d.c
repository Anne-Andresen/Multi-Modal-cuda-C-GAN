#include "conv3d.h"
#include "blas.h"

Conv3D* conv3d_init(int in_channels, int out_channels, int kernel_size, int stride, int padding, int bias) {
    Conv3D* conv = (Conv3D*)malloc(sizeof(Conv3D));
    conv->in_channels = in_channels;
    conv->out_channels = out_channels;
    conv->kernel_size = kernel_size;
    conv->stride = stride;
    conv->padding = padding;
    conv->bias = bias;

    conv->weights = (float*)malloc(sizeof(float) * out_channels * in_channels * kernel_size * kernel_size * kernel_size);
    if (bias) {
        conv->bias_term = (float*)malloc(sizeof(float) * out_channels);
    }

    return conv;
}

void conv3d_free(Conv3D* conv) {
    free(conv->weights);
    if (conv->bias) {
        free(conv->bias_term);
    }
    free(conv);
}

void* conv3d_forward(Conv3D* conv, float* input) {
    int batch_size = input[0];
    int depth = input[1];
    int height = input[2];
    int width = input[3];
    int in_channels = conv->in_channels;

    int out_depth = (depth - conv->kernel_size + 2 * conv->padding) / conv->stride + 1;
    int out_height = (height - conv->kernel_size + 2 * conv->padding) / conv->stride + 1;
    int out_width = (width - conv->kernel_size + 2 * conv->padding) / conv->stride + 1;
    int out_channels = conv->out_channels;

    float* output = (float*)malloc(sizeof(float) * batch_size * out_channels * out_depth * out_height * out_width);

    for (int b = 0; b < batch_size; b++) {
        for (int o_c = 0; o_c < out_channels; o_c++) {
            for (int o_d = 0; o_d < out_depth; o_d++) {
                for (int o_h = 0; o_h < out_height; o_h++) {
                    for (int o_w = 0; o_w < out_width; o_w++) {
                        float sum = 0;
                        for (int i_c = 0; i_c < in_channels; i_c++) {
                            for (int k_d = 0; k_d < conv->kernel_size; k_d++) {
                                for (int k_h = 0; k_h < conv->kernel_size; k_h++) {
                                    for (int k_w = 0; k_w < conv->kernel_size; k_w++) {
                                        int i_d = o_d * conv->stride + k_d - conv->padding;
                                        int i_h = o_h * conv->stride + k_h - conv->padding;
                                        int i_w = o_w * conv->stride + k_w - conv->padding



/*To many nested loops convert ot arithmetic operations on a 1d array lot easier on the GPU */