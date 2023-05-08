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

double xSigmoidFunction (double x)
{
    return 1/(1 + pow(exp(1.0), -1*x));
}

double xTanhFunction(double x) 
{
    return tanh(x);
}

double xReLUFunction(double x) 
{
    return x < 0 ? 0.0 : x;
}

double xLeakeyReLUFunction(double x)
{
    return x < 0 ? 0.1 * x : x;
}

double xELUFunction(double x) 
{
    return x < 0 ? 0.1 * (exp(x) - 1) : x;
}

double xBinaryStepFunction(double x) 
{
    return x < 0 ? 0.0 : 1.0;
}