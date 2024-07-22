#include <conv3d.h>

#define H 128;
#define W 128;
#define D 128;
#define k1 3;
#define k2 3;
#define k3 3;



int main() {
    size_t inputSize = D * H * W * sizeof(float);
    size_t kernelSize = k1 * k2 * k3 * sizeof(float);
    size_t outputSize = D * H * W * sizeof(float);

    // allocate mem host
    float *h_input = (float *)malloc(inputSize);
    float *h_kernel = (float *)malloc(kernelSize);
    float *h_output = (float *)malloc(outputSize);
    // we now actually initialize
    for (int i = 0; i < D * H * W; ++i) h_input[i] = 1.0f; // currently fro trail replace wih image later ofc once we get there ahah
    for (int i =  0; i < k1 * k2 * k3; ++i) h_kernel[i] =  1.0f;
    

    // Init conv structure 
    Conv3D conv;
    conv3d_init(&conv, D, H, W, k1, k2, k3);
    conv3d_set_input(&conv, h_input);
    conv3d_set_kernel(&conv, h_kernel);
    conv3d_execute(&conv, h_output);


    for (int i = 0; i < D * H * W; ++i) {
        if (h_output[i] != k1 * k2 * k3) {
            printf("Error at index %d: %f\n", i, h_output[i]);
            return -1;
        }
    }


printf("Yay all is correct!")

// hastag fredom 
conv3d_free(&conv);
free(h_input);
free(h_kernel);
free(h_output);

reruen 0;

}
