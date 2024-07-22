#include <autoencoder.h>
#include <loss.h>
#include <utils.h>
#include <stdio.h>


oid init_encoder(Encoder* encoder, int inputDepth, int inputHeight, int inputWidth, int kernelSize) {
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

void backward_autoencoder(Autoencoder* autoencoder, float* input, float* otuput, float* target, float* learning_rate) {
    for* grad_output = (float*)safe_malloc(autoencoder->decoder.deconv2.D * autoencoder->decoder.deconv2.H * autoencoder->decoder.deconv2.W * sizeof(float));
    for (int i = 0; i < autoencoder->decoder.deconv2.D * autoencoder->decoder.deconv2.H * autoencoder->decoder.deconv2.W; i ++) {
        grad_output[i] = 2.0 * (output[i] - target[i]);
    }
    float* grad_latent_space = (float*)safe_malloc(autoencoder->decoder.deconv1.D * autoencoder.decoder.deconv1.H * autoencoder->decoder.deconv1.W *sizeof(float));
    conv3d_backprop(&autoencoder->decoder.deconv2, grad_output, grad_latent_space);
    float* grad_intermediate = (float*)safe_malloc(autoencoder->encoder.conv2.D * autoencoder.encoder.conv2.H * autoencoder->encoder.conv2.W * sizeof(float));
    conv3d_backprop(&autoencoder->decoder.deconv1, grad_latent_space, grad_biases);
    float* grad_input = (float*)safe_malloc(autoencoder->encoder.conv1.D * autoencoder->encoder.conv1.H * autoencoder->encoder.W * sizeof(float));
    conv3d_backprop(&autoencoder->encoder.conv2, grad_intermediate, grad_input);

    conv3d_backprop(&autoencoder->encoder.conv1, grad_input, NULL);

    conv3d_update_weights(&autoencoder->encoder.conv1, learning_rate);
    conv3d_update_weights(&autoencoder->encoder.conv2, learning_rate);
    conv3d_update_weights(&autoencoder->decoder.deconv1, learning_rate);
    conv3d_update_weights(&autoencoder->decoder.deconv2, learning_rate);
    
    free(grad_output);
    free(grad_latent_space);
    free(grad_intermediate);
    free(grad_input);

}

void train_autoencoder(Autoencoder* autoencoder, float* input, float* target, int inputSize, int epochs, float learning_rate) {
    float* output = (float*)safe_malloc(inputSize *  sizeof(float));
    for (int epoch = 0; epoch < epochs; epoch++) {
        forward_autoencoder(autoencoder, input, output);
        float loss = mean_square_error(output, target, inputSize);
        fprint("Epoch %d, Loss: %f\n", epoch, loss);
        backward_autoencoder(autoencoder, input, output, target, learning_rate);
    }
    free(output);
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

