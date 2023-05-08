#include "activationFunctions.h"

static const ActivationFunction activationFunction_t[NUM_ACTIVATION_FUNCTIONS] = {
    {.functionType = Sigmoid, .function = xSigmoidFunction},
    {.functionType = Tanh, .function = xTanhFunction},
    {.functionType = ReLU, .function = xReLUFunction},
    {.functionType = LeakeyReLU, .function = xLeakeyReLUFunction},
    {.functionType = ELU, .function = xELUFunction},
    {.functionType = BinaryStep, .function = xBinaryStepFunction},
};

ActivationFunction * xCreateActivationFunction (e_FunctionOption functionType)
{
    ActivationFunction * newActivationFunction = (ActivationFunction *) malloc(sizeof(ActivationFunction));
    newActivationFunction->functionType = functionType;
    newActivationFunction->function = activationFunction_t[functionType].function;

    return newActivationFunction;
}

float xSigmoidFunction (float x)
{
    return 1/(1 + pow(exp(1.0), -1*x));
}

float xTanhFunction(float x) 
{
    return tanh(x);
}

float xReLUFunction(float x) 
{
    return x < 0 ? 0.0 : x;
}

float xLeakeyReLUFunction(float x)
{
    return x < 0 ? 0.1 * x : x;
}

float xELUFunction(float x) 
{
    return x < 0 ? 0.1 * (exp(x) - 1) : x;
}

float xBinaryStepFunction(float x) 
{
    return x < 0 ? 0.0 : 1.0;
}