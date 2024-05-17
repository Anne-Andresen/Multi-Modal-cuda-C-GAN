#include "discriminator.h"
#include "conv3d.h"
#include "batch_norm.h"
#include "leaky_relu.h"
#include "linear.h"

Discriminator* discriminator_init(int input_size) {
    Discriminator* net = (Discriminator*)malloc(sizeof(Discriminator));
    net->input_size = input_size;

    net->conv1 = conv3d_init(input_size, 64, 4, 2, 1, 0);
    net->bn1 = batch_norm_init(64);
    net->leaky_relu1 = leaky_relu_init(0.2);

    net->conv2 = conv3d_init(64, 128, 4, 2, 1, 0);
    net->bn2 = batch_norm_init(128);
    net->leaky_relu2 = leaky_relu_init(0.
