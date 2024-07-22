#include <utils.h>
#include <stdio.h>
#include <stdlib.h>

void* safe_malloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr == NULL) {
        fprint(stderr, "memroy allocation failed");
        exit(EXIT_FAILURE);
    }
    return ptr;
}


void initialize_array(float* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (float)rand() / RAND_MAX
    }
}