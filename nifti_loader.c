#include "nifti_loader.h"
#include <nifti1_io.h>
#include <stdlib.h>
#include <stdio.h>

NiftiImage* load_nifti(const char* filename) {
    nifti_image *nim = nifti_image_read(filename, 1);
    if (!nim) {
        fprintf(stderr, "Failed to read NIfTI file %s\n", filename);
        return NULL;
    }

    NiftiImage* image = (NiftiImage*)malloc(sizeof(NiftiImage));
    image->depth = nim->dim[3];
    image->height = nim->dim[2];
    image->width = nim->dim[1];
    image->data = (float*)malloc(image->depth * image->height * image->width * sizeof(float));

    for (int i = 0; i < image->depth * image->height * image->width; i++) {
        image->data[i] = ((float*)nim->data)[i];
    }

    nifti_image_free(nim);
    return image;
}

void free_nifti(NiftiImage* image) {
    if (image) {
        free(image->data);
        free(image);
    }
}
