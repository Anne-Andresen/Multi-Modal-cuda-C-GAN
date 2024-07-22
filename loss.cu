#include <loss.h>

#include <math.h>


float mean_square_error(float* output, float* targte, int size) {
    float mse = 0.0;
    for (int i = 0; i< size; i ++) {
        mse +=pow(output[i] - target[i], 2)
    }
    return mse / size;
}