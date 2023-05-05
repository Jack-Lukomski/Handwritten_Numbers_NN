#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../Matrix/matrix.h"

typedef struct {
    uint16_t numInputs;
    Matrix * inputLayer;
} InputLayer;

typedef struct {
    Matrix * biases;
    Matrix * hiddenLayer;
} HiddenLayer;

typedef struct {
    uint16_t numOutputs;
    float * biases;
    Matrix outputLayer;
} OutputLayer;

typedef struct
{
    InputLayer inputLayer;
    HiddenLayer * hiddenLayers;
    OutputLayer outputLayer;
} NerualNetwork;

InputLayer * xConstuctInputLayer (uint16_t numInputs, Matrix * inputMatrix);
HiddenLayer * xConstructHiddenLayers (uint16_t numLayers, Matrix * hiddenLayerMatrixs, Matrix * biases);


#endif