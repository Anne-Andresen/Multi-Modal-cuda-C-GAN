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

void backward()