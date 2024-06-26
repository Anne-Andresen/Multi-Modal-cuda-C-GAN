
/*
#ifndef CONV3D_H
#define CONV3D_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int bias;
    float* weights;
    float* bias_term;
} Conv3D;

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

    // Xavier initialization of weights
    if (bias) {
        for (int i = 0; i < out_channels * in_channels * kernel_size * kernel_size * kernel_size; i++) {
            conv->weights[i] = (float) rand() / RAND_MAX * 2 * sqrt(6.0 / (in_channels * kernel_size * kernel_size * kernel_size + out_channels));
        }
        for (int i = 0; i < out_channels; i++) {
            conv->bias_term[i] = 0;
        }
    } else {
        for (int i = 0; i < out_channels * in_channels * kernel_size * kernel_size * kernel_size; i++) {
            conv->weights[i] = (float) rand() / RAND_MAX * 2 * sqrt(6.0 / (in_channels * kernel_size * kernel_size * kernel_size));
        }
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


*/

#include <conv3d.h>
#include <stdlib.h>
#include <stdio.h>


__global__ void conv3d_kernel(const float* input, const float* kernel, float* output, int D, int H, int W, int k1, int k2, int k3){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int pad_D = k1 / 2;
    int pad_H = k2 / 2;
    int pad_W = k3 / 2;

    if (x >= W || y >= H || z >= D) return;

    float sum = 0.0f;

    
}