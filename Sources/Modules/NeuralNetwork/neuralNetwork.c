#include "neuralNetwork.h"

static InputLayer * xConstuctInputLayer (Matrix * inputMatrix)
{
    InputLayer * newInputLayer = (InputLayer *) malloc(sizeof(InputLayer));
    newInputLayer->inputLayer = inputMatrix;

    return newInputLayer;
}

static HiddenLayer * xConstructHiddenLayer (Matrix * hiddenLayerMatrix, Matrix * biases)
{
    HiddenLayer * newHiddenLayer = (HiddenLayer *) malloc(sizeof(HiddenLayer));
    newHiddenLayer->biases = biases;
    newHiddenLayer->hiddenLayer = hiddenLayerMatrix;

    return newHiddenLayer;
}

static OutputLayer * xConstructOutputLayer (Matrix * outputLayerMatrix, Matrix * biases)
{
    OutputLayer * newOutputLayer = (OutputLayer *) malloc(sizeof(OutputLayer));
    newOutputLayer->biases = biases;
    newOutputLayer->outputLayer = outputLayerMatrix;

    return newOutputLayer;
}

NerualNetwork * xConstructNeuralNetwork (Matrix * inputMatrix, uint16_t numHiddenLayers, Matrix * hiddenLayerMatricies[numHiddenLayers], Matrix * hiddenLayerBiases[numHiddenLayers], Matrix * outputLayerMatrix, Matrix * outputLayerBiases)
{
    NerualNetwork * newNN = (NerualNetwork *) malloc(sizeof(NerualNetwork));

    newNN->inputLayer = xConstuctInputLayer(inputMatrix);
    newNN->outputLayer = xConstructOutputLayer(outputLayerMatrix, outputLayerBiases);
    newNN->numHiddenLayers = numHiddenLayers;
    newNN->hiddenLayers = (HiddenLayer **) malloc(numHiddenLayers * sizeof(HiddenLayer *));

    for (uint16_t currHL = 0; currHL < numHiddenLayers; currHL++)
    {
        newNN->hiddenLayers[currHL] = xConstructHiddenLayer(hiddenLayerMatricies[currHL], hiddenLayerBiases[currHL]);
    }

    return newNN;
}