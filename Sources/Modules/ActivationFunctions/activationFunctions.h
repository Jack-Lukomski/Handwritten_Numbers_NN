#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define NUM_ACTIVATION_FUNCTIONS 6

typedef double (*xActivationFunctionPtr)(double x);

double xSigmoidFunction (double x);
double xTanhFunction (double x);
double xReLUFunction (double x);
double xLeakeyReLUFunction (double x);
double xELUFunction (double x);
double xBinaryStepFunction (double x);

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