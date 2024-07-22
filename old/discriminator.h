#ifndef DISCRIMINATOR_H
#define DISCRIMINATOR_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int input_size;
    void* conv1;
    void* bn1;
    void* leaky_relu1;
    void* conv2;
    void* bn2;
    void* leaky_relu2;
    void* conv3;
    void* bn3;
    void* leaky_relu3;
    void* conv4;
    void* bn4;
    void* leaky_relu4;
    void* fc;
} Discriminator;

Discriminator* discriminator_init(int input_size);
void discriminator_free(Discriminator* net);
void* discriminator_forward(Discriminator* net, float* x);

#endif  // DISCRIMINATOR_H
