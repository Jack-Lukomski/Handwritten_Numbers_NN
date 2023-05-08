#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../Matrix/matrix.h"
#include "../ActivationFunctions/activationFunctions.h"

typedef struct {
    uint16_t numInputs;
    Matrix * inputLayer;
} InputLayer;

typedef struct {
    Matrix * biases;
    Matrix * hiddenLayer;
} HiddenLayer;

typedef struct {
    Matrix * biases;
    Matrix * outputLayer;
} OutputLayer;

typedef struct
{
    InputLayer * inputLayer;
    uint16_t numHiddenLayers;
    HiddenLayer ** hiddenLayers;
    OutputLayer * outputLayer;
} NerualNetwork;

static InputLayer * xConstuctInputLayer (Matrix * inputMatrix);
static HiddenLayer * xConstructHiddenLayer (Matrix * hiddenLayerMatrix, Matrix * biases);
static OutputLayer * xConstructOutputLayer (Matrix * outputLayerMatrix, Matrix * biases);
NerualNetwork * xConstructNeuralNetwork (Matrix * inputMatrix, uint16_t numHiddenLayers, Matrix * hiddenLayerMatricies[numHiddenLayers], Matrix * hiddenLayerBiases[numHiddenLayers], Matrix * outputLayerMatrix, Matrix * outputLayerBiases);
Matrix * xComputeOutputSums (NerualNetwork * NN, e_FunctionOption activationFunction);
void vPrintAllLayers (NerualNetwork * NN);

#endif