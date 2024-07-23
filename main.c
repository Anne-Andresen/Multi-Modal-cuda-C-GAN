#include "autoencoder.h"
#include "utils.h"
#include "nifti_loader.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <nifti_file>\n", argv[0]);
        return 1;
    }

    const char* nifti_file = argv[1];
    NiftiImage* image = load_nifti(nifti_file);
    if (!image) {
        return 1;
    }

    int inputDepth = image->depth, inputHeight = image->height, inputWidth = image->width;
    int kernelSize = 3;
    float* inputData = image->data;
    float* targetData = (float*)safe_malloc(inputDepth * inputHeight * inputWidth * sizeof(float));
    
    // Use the input image as the target for simplicity
    for (int i = 0; i < inputDepth * inputHeight * inputWidth; i++) {
        targetData[i] = inputData[i];
    }

    Autoencoder autoencoder;
    init_autoencoder(&autoencoder, inputDepth, inputHeight, inputWidth, kernelSize);

    int epochs = 100;
    float learning_rate = 0.01;
    train_autoencoder(&autoencoder, inputData, targetData, inputDepth * inputHeight * inputWidth, epochs, learning_rate);

    free_autoencoder(&autoencoder);
    free_nifti(image);
    free(targetData);

    return 0;
}
