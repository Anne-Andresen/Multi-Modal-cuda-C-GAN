#ifndef LEAKY_RELU_H
#define LEAKY_RELU_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float alpha;
} LeakyRelU;

LeakyRelU* leaky_relu_init(float alpha);
void leaky_relu_free(LeakyRelU* leaky_relu);
void* leaky_relu_forward(LeakyRelU* leaky_relu, float* input);

#endif  // LEAKY_RELU_H