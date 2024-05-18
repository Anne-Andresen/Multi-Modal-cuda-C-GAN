#ifndef CONV3D_H
#define CONV3D_H

#include <stdio.h>
#include <stdlib.h>

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

Conv3D* conv3d_init(int in_channels, int out_channels, int kernel_size, int stride, int padding, int bias);
void conv3d_free(Conv3D* conv);
void* conv3d_forward(Conv3D* conv, float* input);

#endif  // CONV3D_H