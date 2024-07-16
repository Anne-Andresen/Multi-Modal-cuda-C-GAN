#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "propagation.h"

/* reg array
void forward(float input[INPUT_SIZE], float weights[INPUT_SIZE][OUTPUT_SIZE], float output[OUTPUT_SIZE]) {
    for (int i = 0; i < OUTPUT_SIZE; i ++) {
        output[i] = 0.0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            output[i] + = input[j] * weights[j][i];
        }
        output[i] = fmax(0.0, output[i]);
    }
}

void backward(float input[INPUT_SIZE], float weights[INPUT_SIZE][OUTPUT_SIZE], float output[OUTPUT_SIZE], float target[OUTPUT_SIZE], float dW[INPUT_SIZE][OUTPUT_SIZE]) {
    float dOutput[OUTPUT_SIZE];

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        dOutput[i] = output[i] - target[i]; very simple loss for now to be replaced later 111209
        if (output[i] <= 0) dOutput = 0; // derivative of relu

        for (int j = 0; i < INPUT_SIZE; j++) {
            dW[j][i] = dOutput[i] * input[j];

        }
    }
}


In the follwing we are very inspired by Adam 111209
void update_weights(float weights[INPUT_SIZE][OUTPUT_SIZE], float dW[INPUT_SIZE][OUTPUT_SIZE], float m[INPUT_SIZE][OUTPUT_SIZE], float v[INPUT_SIZE][OUTPUT_SIZE], int t) {
    const float beta1 = 0.9;
    const float beta2 = 0.999;
    const float epsilon = 1e-8;

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (j = 0; j < OUTPUT_SIZE; j ++) {
            m[i][j] = beta1 * m[i][j] / (1.0 - beta1) * dW[i][j];
            v[i][j] = beta2 * m[i][j] / (1.0 - beta2) * dW[i][j] * dW[i][j];

            float m_hat = m[i][j] / (1 - pow(beta1, t));
            float v_hat = v[i][j] / (1 - pow(beta2, t));

            weights[i][j] -= LEARNING_RATE * m_hat / (sqrt(v_hat) + epsilon);
        }
    }

}
*/

// Forward propagation
void forward(float input[INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float weights[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH], float output[OUTPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]) {
    // Implement forward propagation logic for 3D images
    // This will typically involve convolution operations
}

// Backpropagation
void backward(float input[INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float weights[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH], float output[OUTPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float target[OUTPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float dW[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH]) {
    // Implement backpropagation logic for 3D images
}

// Update weights using Adam optimizer
void update_weights(float weights[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH], float dW[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH], float m[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH], float v[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH], int t) {
    const float beta1 = 0.9;
    const float beta2 = 0.999;
    const float epsilon = 1e-8;

    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            for (int k = 0; k < KERNEL_SIZE; k++) {
                for (int d = 0; d < INPUT_DEPTH; d++) {
                    for (int o = 0; o < OUTPUT_DEPTH; o++) {
                        m[i][j][k][d][o] = beta1 * m[i][j][k][d][o] + (1.0 - beta1) * dW[i][j][k][d][o];
                        v[i][j][k][d][o] = beta2 * v[i][j][k][d][o] + (1.0 - beta2) * dW[i][j][k][d][o] * dW[i][j][k][d][o];

                        float m_hat = m[i][j][k][d][o] / (1.0 - pow(beta1, t));
                        float v_hat = v[i][j][k][d][o] / (1.0 - pow(beta2, t));

                        weights[i][j][k][d][o] -= LEARNING_RATE * m_hat / (sqrt(v_hat) + epsilon);
                    }
                }
            }
        }
    }
}
/*CUDA KERNELS*/


__global__ void forward_kernel(float *input, float *weights, float *output) {
    // up coming 
}


__global__ void backward_kernel(float *input, float *weights, float *output, float *target, float *dW) {
    // up coming 100899
}

__global__ void update_weights_kernel(float *weights, float *dW, float *m. float *v, int t) {
    const float beta1 = 0.9;
    const float beta2 = 0.999;
    const float epsilon = 1e-8;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE * INPUT_DEPTH * OUTPUT_DEPTH) {
        m[idx] = beta1 * m[idx] + (1.0 - beta1) * dW[idx];
        v[idx] = beta2 * v[idx] + (1.0 - beta2) * dW[idx] * dW[idx];

        float m_hat[idx] = m[idx] / (1.0 - powf(beta1, t));
        float v_hat[idx] = v[idx] / (1.0 - powf(beta2, 1));

        weights[idx] -= LEARNING_RATE * m_hat / (sqrtf(v_hat) + epsilon) 
    }
}

__global__ attention_forward(float* input, float* output, int depth, int height, int width) {
    // attention forward propagation comming up

}


__global__ conv_forward(float* input, float* output, int depth, int height, int width){
    // conv forward propagation comming up

}


__global__ attention_forward_kernel(float* input, float* output, int depth, int height, int width) {
    // cuda attention forward propagation comming up

}

__global__ conv_forward_kernel(float* input, float* output, int depth, int height, int width) {
    // cuda conv forward propagation comming up

}
int main() {
    float input[INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];  // Initialize input data
    float target[OUTPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]; // Initialize target data
    float weights[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH]; // Initialize weights
    float output[OUTPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]; // Initialize output buffer

    float dW[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH] = {0};
    float m[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH] = {0};
    float v[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH] = {0};

    float *d_input, *d_weights, *d_output, *d_target, *d_dW, *d_m, *d_v;
    cudaMalloc((void**)&d_input, INPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float));
    cudaMalloc((void**)&d_weights, KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE * INPUT_DEPTH * OUTPUT_DEPTH * sizeof(float));
    cudaMalloc((void**)&d_output, OUTPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float));
    cudaMalloc((void**)&d_target, OUTPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float));
    cudaMalloc((void**)&d_dW, KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE * INPUT_DEPTH * OUTPUT_DEPTH * sizeof(float));
    cudaMalloc((void**)&d_m, KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE * INPUT_DEPTH * OUTPUT_DEPTH * sizeof(float));
    cudaMalloc((void**)&d_v, KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE * INPUT_DEPTH * OUTPUT_DEPTH * sizeof(float));

    cudaMemcpy(d_input, input, INPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE * INPUT_DEPTH * OUTPUT_DEPTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, OUTPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    int epochs = 1000;
    for (int t = 1; t <= epochs; t++) {
        forward_kernel<<<1, OUTPUT_DEPTH>>>(d_input, d_weights, d_output);
        cudaDeviceSynchronize();
        cudaMemcpy(output, d_output, OUTPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

        backward_kernel<<<1, OUTPUT_DEPTH>>>(d_input, d_weights, d_output, d_target, d_dW);
        cudaDeviceSynchronize();

        update_weights_kernel<<<1, KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE * INPUT_DEPTH * OUTPUT_DEPTH>>>(d_weights, d_dW, d_m, d_v, t);
        cudaDeviceSynchronize();

        if (t % 100 == 0) {
            printf("Epoch %d: Output: %f %f\n", t, output[0][0][0], output[0][0][1]);
        }
    }

    cudaMemcpy(weights, d_weights, KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE * INPUT_DEPTH * OUTPUT_DEPTH * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    cudaFree(d_target);
    cudaFree(d_dW);
    cudaFree(d_m);
    cudaFree(d_v);

    return 0;
}