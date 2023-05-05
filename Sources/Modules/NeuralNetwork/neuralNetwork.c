#include "neuralNetwork.h"

InputLayer * xConstuctInputLayer (uint16_t numInputs, Matrix * inputMatrix)
{
    InputLayer * newInputLayer = (InputLayer *) malloc(sizeof(InputLayer));
    newInputLayer->numInputs = numInputs;
    newInputLayer->inputLayer = inputMatrix;

    return newInputLayer;
}

HiddenLayer * xConstructHiddenLayer (Matrix * hiddenLayerMatrix, Matrix * biases)
{
    HiddenLayer * newHiddenLayers = (HiddenLayer *) malloc(sizeof(Matrix));
    newHiddenLayers->biases = biases;
    newHiddenLayers->hiddenLayer = hiddenLayerMatrix;

    return newHiddenLayers;
}