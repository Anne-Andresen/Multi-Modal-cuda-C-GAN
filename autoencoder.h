#ifndef AUTOENCODER_H
#define AUTOENCODER_H
#include "conv3.h"

typedef struct {
    Conv3D conv1;
    Conv3D conv2;
} Encoder;

typedef struct {
    Conv3D deconv1;
    Conv3D deconv2;
} Decoder;

typedef struct {
    Encoder encoder;
    Decoder decoder; 
} Autoencoder;


void init_encoder(Encoder* encoder, int inputDepth, int inputHeight, int inputWidth, int kernelSize);
void init_decoder(Decoder* decoder, int inputDepth, int inputHeight, int inputWidth, int kernelSize);
void init_autoencoder(Autoencoder* autoencoder, int inputDepth, int inputHeight, int inputWidth, int kernelSize);

void forward_encoder(Encoder* encoder, float* input, float* output);
void forward_decoder(Decoder* decoder, float* input, float* output);
void forward_autoencoder(Autoencoder* autoencoder, float* input, float* output);

void backward_autoencoder(Autoencoder* autoencoder, float* input, float* output, float* target, float learning_rate);
void train_autoencoder(Autoencoder* autoencoder, float* input, float* output, float* target, float* inputSize, int epoch, float learning_rate);



void free_encoder(Encoder* encoder);
void free_decoder(Decoder* decoder);
void free_autoencoder(Autoencoder* autoencoder);

#endif


