#ifndef LINEAR_H
#define LINEAR_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int in_features;
    int out_features;
    float* weights;
    float* bias;
} Linear;

Linear* linear_init(int in_features, int out_features);
void linear_free(Linear* linear);
void* linear_forward(Linear* linear, float* input);

#endif  // LINEAR_H