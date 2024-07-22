#ifndef UNET3D_H
#define UNET3D_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int in_channels;
    int out_channels;
    int embed_dim;
    int num_heads;
    void* attention;
    void* encoder;
    void* decoder;
    void* final_conv;
} UNet3D;

UNet3D* unet3d_init(int in_channels, int out_channels);
void unet3d_free(UNet3D* net);
void* unet3d_forward(UNet3D* net, float* x, float* struct);

#endif  // UNET3D_H
