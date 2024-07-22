#include "unet3d.h"
#include "cross_attention.h"
#include "conv3d.h"
#include "relu.h"
#include "max_pool3d.h"
#include "conv_transpose3d.h"

UNet3D* unet3d_init(int in_channels, int out_channels) {
    UNet3D* net = (UNet3D*)malloc(sizeof(UNet3D));
    net->in_channels = in_channels;
    net->out_channels = out_channels;
    net->embed_dim = 128;
    net->num_heads = 8;

    net->attention = cross_attention_init(net->embed_dim, net->num_heads);

    // Encoder layers
    net->encoder = conv3d_init(in_channels, 64, 3, 1);
    net->encoder = relu_init(net->encoder);
    net->encoder = conv3d_init(64, 64, 3, 1);
    net->encoder = relu_init(net->encoder);
    net->encoder = max_pool3d_init(2, 2);

    // Decoder layers
    net->decoder = conv_transpose3d_init(64, 64, 2, 2);
    net->decoder = relu_init(net->decoder);
    net->decoder = conv3d_init(64, 64, 3, 1);
    net->decoder = relu_init(net->decoder);
    net->decoder = conv3d_init(64, out_channels, 1);

    net->final_conv = conv3d_init(in_channels, out_channels, 1);

    return net;
}

void unet3d_free(UNet3D* net) {
    cross_attention_free(net->attention);
    conv3d_free(net->encoder);
    conv_transpose3d_free(net->decoder);
    conv3d_free(net->final_conv);
    free(net);
}

void* unet3d_forward(UNet3D* net, float* x, float* struct) {
    x = cross_attention_forward(net->attention, x, struct, struct);
    x = conv3d_forward(net->encoder, x);
    x = conv_transpose3d_forward(net->decoder, x);
    return conv3d_forward(net->final_conv, x);
}
