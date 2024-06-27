#ifndef CONV3D_H
#define CONV3D_H

#include <cuda_runtime.h>

typedef struct {

   int D, H, W;
   int k1, k2, k3;
   float *device_input;
   float *device_kernel;
   float *device_output;

} Conv3D;

void conv3d_init(Conv3D*conv, int input_depth, int input_height, int input_widt, int kernel_depth, int kernel_height, int kernel_width);
void conv3d_set_input(Conv3D* conv, const float* input_data);
void conv3d_set_kernel(Conv3D* conv, const float* kernel_data);
void conv3d_execute(Conv3D* conv, float* output_data);
void conv3d_free(Conv3D* conv);
#endif 