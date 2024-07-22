#include <stdio.h>
#include <stdlib.h>
#include <conv3d.h>


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

void init_encoder(Encoder* encoder, int inputDepth, int inputHeight, int inputWidth, int kernelSize) {
    conv3d_init(&encoder->conv1, inputDepth, inputHeight, inputWidth, kernelSize, kernelSize, kernelSize);
    conv3d_init(&encoder->conv2, inputDepth, inputHeight, inputWidth, kernelSize, kernelSize, kernelSize);
    
}

void init_decoder(Decoder* decoder, int inputDepth, int inputHeight, int inputWidth, int kernelSize) {
    conv3d_init(&decoder->deconv1, inputDepth, inputHeight, inputWidth, kernelSize, kernelSize, kernelSize)
    conv3d_init(&decoder->deconv2, inputDepth, inputHeight, inputWidth, kernelSize, kernelSize, kernelSize)

}
/*include additonal lernel size for down sampling*/
void init_autoencoder(Autoencoder* autoencoder, int inputDepth, int inputHeight, int inputWidth, int kernelSize) {
    init_encoder(&autoencoder->encoder, inputDepth, inputHeight, inputWidth, kernelSize);
    init_decoder(&autoencoder->decoder, inputDepth, inputHeight, inputWidth, kernelSize);
}

void forward_encoder(Encoder* encoder, float* input, float*output) {
    float* inter_output = (float*)malloc(encoder->conv1.D * encoder->conv1.H * encoder->conv1.W * sizeof(float));
    conv3d_set_input(&encoder->conv1, input);
    conv3d_execute(&encoder->conv1, inter_output);

    conv3d_set_input(&encoder->conv2, inter_output);
    conv3d_execute(&encoder->conv2, output);

    free(inter_output);

}

void forward_decoder(Decoder* decoder, float* input, float* output) {
    float* inter_output = (float*)malloc(decoder->deconv1.D * decoder->deconv1.H * decoder->deconv1.W * sizeof(float));
    conv3d_init(&decoder->deconv1, input); //mssing upsample here 
    conv3d_execute(&decoder->deconv1, inter_output);

    conv3d_init(&decoder->deconv2, inter_output);
    conv3d_execute(&decoder->deconv2, output);

    free(inter_output);
}


void forward_autoencoder(Autoencoder* autoencoder, float* input, float* output) {
    float* latent_space = (float*)malloc(encoder->conv2.D * encoder->conv2.H * encoder->conv2.W * sizeof(float));
    forward_encoder(&autoencoder->encoder, input, latent_space);
    forward_decoder(&autoencoder->decoder, latent_space, output);
    free(latent_space);

}

void free_encoder(Encoder* encoder) {
    conv3d_free(&encoder->conv1);
    conv3d_free(&encoder->conv2);
}


void free_decoder(Decoder* decoder) {
    conv3d_free(&decoder->deconv1);
    conv3d_free(&decoder->deconv2);
}

void free_autoencoder(Autoencoder* autoencoder) {
    free_encoder(&autoencoder->encoder);
    free_decoder(&autoencoder->decoder);
}


int main() {
    int inputDepth = 32, inputHeight = 32, inputWidth = 32;
    int kernelSize = 3;
    float* inputData = (float*)malloc(inputDepth * inputHeight * inputWidth * sizeof(float));
    float* outptData = (float*)malloc(inputDepth * inputHeight * inputWidth * sizeof(float));

    Autoencoder autoencoder;
    init_autoencoder(&autoencoder, inputDepth, inputHeight, inputWidth, kernelSize);
    forward_autoencoder(&autoencoder, inputData, outptData);
    free_autoencoder(&autoencoder);
    free(inputData);
    free(outptData);

    return 0;

}
