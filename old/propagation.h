#ifndef PROPAGATION_H
#define PROPAGATION_H

#define INPUT_DEPTH 3
#define INPUT_HEIGHT 32
#define INPUT_WIDTH 32
#define OUTPUT_DEPTH 64
#define KERNEL_SIZE 3
#define LEARNING_RATE 0.0002



void forward(float input[INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float weights[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH], float output[OUTPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);
void backward(float input[INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float weights[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH], float output[OUTPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float target[OUTPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float dW[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH]);
void update_weights(float weights[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH], float dW[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH], float m[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH], float v[KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE][INPUT_DEPTH][OUTPUT_DEPTH], int t);


__global__ void forward_kernel(float *input, float *weights, float *output);
__global__ void backward_kernel(float *input, float *weights, float *output, float *target, float *dW);
__global__ void update_weights_kernel(float *weights, float *dW, float *m, float *v, int t);



void attention_forward(float *intput, float* output, int depth, int height, int width);
void conv_forward(float* input, float* output, int depth, int height, int width);

__global__ void attention_forward_kernel(float* input, float* output, int depth, int height, int width);
__global__ void conv_forward_kernel(float* input, float* output, int, depth, int height, int width);



#endif