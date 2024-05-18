#ifndef BATCH_NORM_H
#define BATCH_NORM_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float* gamma;
    float* beta;
    float* running_mean;
    float* running_var;
    float epsilon;
    int size;
} BatchNorm;

BatchNorm* batch_norm_init(int size);
void batch_norm_free(BatchNorm* bn);
void* batch_norm_forward(BatchNorm* bn, float* input);

#endif  // BATCH_NORM_H