#ifndef CROSS_ATTENTION_H
#define CROSS_ATTENTION_H

#include <cuda_runtime.h>


typedef struct {
    int embed_dim;
    int num_heads;
    int head_dim;

}
