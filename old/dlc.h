#ifndef DL_COMMON_H
#define DL_COMMON_H

#include <stdint.h>

typedef int64_t DLTensorIndex;
typedef float DLFloat;

typedef enum {
    DL_device_cpu,
    DL_device_gpu,
} DLDeviceType;

typedef struct {
    DLDeviceType type;
    int index;
} DLDevice;

typedef struct {
    DLTensor *tensors;
    int size;
} DLTensorList;

typedef enum {
    DL_loss_mse,
    DL_loss_binary_crossentropy,
    DL_loss_categorical_crossentropy,
} DLLossType;

typedef struct {
    DLLossType type;
} DLC_Loss;

typedef enum {
    DL_optimizer_sgd,
    DL_optimizer_adam,
} DLOptimizerType;

typedef struct {
    DLOptimizerType type;
} DLC_Optimizer;

#endif // DL_COMMON_H

#ifndef DL_MODEL_H
#define DL_MODEL_H

typedef struct {
    // Model parameters and weights
} DLC_Model;

DLDevice DLC_ModelCreateDevice(DLDevice device);
DLC_Model DLC_ModelCreate(...);
void DLC_ModelSetDevice(DLC_Model *model, DLDevice device);
void DLC_ModelForward(DLC_Model *model, DLTensor *input, DLTensor *input_2, DLTensor *output);
void DLC_ModelBackward(DLC_Model *model);
void DLC_ModelClearGradients(DLC_Model *model);

#endif // DL_MODEL_H

#ifndef DL_LOSS_H
#define DL_LOSS_H

DLC_Loss DLC_LossCreate(DLLossType type);
void DLC_LossCompute(DLC_Loss *loss, DLTensor *y_true, DLTensor *y_pred, DLTensor *loss_value);

#endif // DL_LOSS_H

#ifndef DL_OPTIMIZER_H
#define DL_OPTIMIZER_H

DLC_Optimizer DLC_OptimizerCreate(DLOptimizerType type);
void DLC_OptimizerStep(DLC_Optimizer *optimizer);

#endif // DL_OPTIMIZER_H
