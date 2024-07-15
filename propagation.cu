#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "propagation.h"


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
        dOutput[i] = output[i] - target[i]; /*very simple loss for now to be replaced later 111209*/
        if (output[i] <= 0) dOutput = 0; // derivative of relu

        for (int j = 0; i < INPUT_SIZE; j++) {
            dW[j][i] = dOutput[i] * input[j];

        }
    }
}



