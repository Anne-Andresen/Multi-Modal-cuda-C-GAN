#ifndef NIFTI_LOADER_H
#define NIFTI_LOADER_H

typedef struct {
    int depth;
    int height;
    int width;
    float* data;
} NiftiImage;

NiftiImage* load_nifti(const char* filename);
void free_nifti(NiftiImage* image);

#endif // NIFTI_LOADER_H
