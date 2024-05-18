#ifndef CONV_TRANSPOSE3D_H
#define c

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int output_padding;
    float* weights;
    float* bias;
} ConvTranspose3D;

ConvTranspose3D* conv_transpose3d_init(int in_channels, int out_channels, int kernel_size, int stride, int padding, int output_padding);
void conv_transpose3d_free(ConvTranspose3D* conv_transpose3d);
void* conv_transpose3d_forward(ConvTranspose3D* conv_transpose3d, float* input);

#endif  // CONV_TRANSPOSE3D_H