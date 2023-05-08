#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define NUM_ACTIVATION_FUNCTIONS 6

typedef float (*xActivationFunctionPtr)(float x);

float xSigmoidFunction (float x);
float xTanhFunction (float x);
float xReLUFunction (float x);
float xLeakeyReLUFunction (float x);
float xELUFunction (float x);
float xBinaryStepFunction (float x);

typedef enum {
    Sigmoid,
    Tanh,
    ReLU,
    LeakeyReLU,
    ELU,
    BinaryStep,
} e_FunctionOption;

typedef struct {
    e_FunctionOption functionType;
    xActivationFunctionPtr function;
} ActivationFunction;

ActivationFunction * xCreateActivationFunction (e_FunctionOption functionType);

#endif