
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

    for (int K1 = 0; K1 < k1; K1++) {
        for (int K2 = 0; K2 < k2; K2++) {
            for (int K3=0; K3 < k3; K3++) {
                int inZ = z + K1 - pad_D;
                int inY = y + K2 - pad_H;
                int inX = x + K3 - pad_W;

                if (inZ >= 0 && inZ < D && inY >= 0 && inY < H && inX >=0 && inX < W) {
                    sum += input[inZ * H * W + inY * W + inX] * kernel[K1 * k3 * k2 + K2 * k3 + K3];

                }
            }
        }
    }
    output[z * H * W  + y * W + x] = sum;

}

void conv3d_init(Conv3D* conv, int inputDepth, int inputHeight, int inputWidth, int kernelDepth, int kernelHeight, int kernelWidth) {
    conv->D = inputDepth;
    conv->H = inputHeight;
    conv->W = inputWidth;

    conv->k1 = kernelDepth;
    conv->k2 = kernelHeight;
    conv->k3 = kernelWidth;

    size_t inputSize = conv->D * conv->H * conv->W * sizeof(float);
    size_t kernelSize = conv->k1 * conv->k2 * conv->k3 * sizeof(float);
    size_t outputSize = conv->D * conv->H * conv->W * sizeof(float);

    cudaMalloc(&conv->device_input, inputSize);
    cudaMalloc(&conv->device_kernel, kernelSize);
    cudaMalloc(&conv->device_output, outputSize);

}


void conv3d_set_input(Conv3D* conv, const float* inputData) {
    size_t inputSize = conv->D * conv->H * conv->W *sizeof(float);
    cudaMemcpy(conv->device_input, inputData, inputSize, cudaMemcpyHostToDevice);
}


void conv3d_set_kernel(Conv3D* conv, const float* kernelData) {
    size_t kernelData = conv->k1 * conv->k2 * conv->k3 * sizeof(float);
    cudaMemcpy(conv->device_kernel, kernelData, kernelSize, cudaMemcpyHostToDevice);
}

void conv3d_execute(Conv3D* conv, float* outputData) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((conv->W + blockSize.x - 1) / blockSize.x, (conv->H + blockSize.y - 1) / blockSize.y, (conv->D + blockSize.z - 1) / blockSize.z);

    conv3d_kernel<<<gridSize, blockSize>>>(conv->device_input, conv->device_kernel, conv->device_output, conv->D, conv->H, conv->W, conv->k1, conv->);
    cudaDeviceSynchronize();

    size_t outputSize = conv->D * conv ->H * conv->W * sizeof(float);
    cudaMemcpy(outputData, conv-device_output, outputSize, cudaMemcpyDevicetoHost);
}

void conv3d_free(Conv3D* conv) {
    cudaFree(conv->device_input);
    cudaFree(conv->device_kernel);
    cudaFree(conv->device_output;)
}

extern "C" void launch_conv3d_kerneæ(const float* d_input, const float* d_kernel, float* output, int D, int H, int W, int k1, int k2, int k3) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((W + blockSize.x -1) / blockSize.x, (H + blockSize.y - 1) / blockSize.y, (D + blockSize.z - 1) / blockSize.z);

    conv3d_kernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, D, H, W, k1, k2, k3);
    cudaDeviceSynchronize();
}
*/


#include "conv3d.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void conv3d_kernel(float* input, float* weights, float* biases, float* output, int D, int H, int W, int kD, int kH, int kW) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && z < D) {
        float value = biases[0]; // Assuming a single bias value for simplicity
        for (int kd = 0; kd < kD; kd++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int in_d = z - kd + kD / 2;
                    int in_h = y - kh + kH / 2;
                    int in_w = x - kw + kW / 2;
                    if (in_d >= 0 && in_d < D && in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                        value += input[(in_d * H + in_h) * W + in_w] * weights[(kd * kH + kh) * kW + kw];
                    }
                }
            }
        }
        output[(z * H + y) * W + x] = value;
    }
}

void conv3d_init(Conv3D* conv, int inputDepth, int inputHeight, int inputWidth, int kernelD, int kernelH, int kernelW) {
    conv->D = inputDepth;
    conv->H = inputHeight;
    conv->W = inputWidth;
    conv->kernelD = kernelD;
    conv->kernelH = kernelH;
    conv->kernelW = kernelW;
    conv->weights = (float*)malloc(kernelD * kernelH * kernelW * sizeof(float));
    conv->biases = (float*)malloc(sizeof(float));
    conv->grad_weights = (float*)malloc(kernelD * kernelH * kernelW * sizeof(float));
    conv->grad_biases = (float*)malloc(sizeof(float));
    // Initialize weights and biases
    for (int i = 0; i < kernelD * kernelH * kernelW; i++) {
        conv->weights[i] = (float)rand() / RAND_MAX;
    }
    conv->biases[0] = (float)rand() / RAND_MAX;
}

void conv3d_set_input(Conv3D* conv, float* input) {
    conv->input = input;
}

void conv3d_execute(Conv3D* conv, float* output) {
    conv->output = output;
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((conv->W + blockDim.x - 1) / blockDim.x, (conv->H + blockDim.y - 1) / blockDim.y, (conv->D + blockDim.z - 1) / blockDim.z);
    conv3d_kernel<<<gridDim, blockDim>>>(conv->input, conv->weights, conv->biases, conv->output, conv->D, conv->H, conv->W, conv->kernelD, conv->kernelH, conv->kernelW);
}

void conv3d_backprop(Conv3D* conv, float* grad_output, float* grad_input) {
    // Implement backpropagation for convolution
}

void conv3d_update_weights(Conv3D* conv, float learning_rate) {
    for (int i = 0; i < conv->kernelD * conv->kernelH * conv->kernelW; i++) {
        conv->weights[i] -= learning_rate * conv->grad_weights[i];
    }
    conv->biases[0] -= learning_rate * conv->grad_biases[0];
}

void conv3d_free(Conv3D* conv) {
    free(conv->weights);
    free(conv->biases);
    free(conv->grad_weights);
    free(conv->grad_biases);
}
