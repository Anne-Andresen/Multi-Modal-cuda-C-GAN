#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "propagation.h"

/*
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


In the follwing we are very inspired by Adam 10899
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

