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

void vPrintAllLayers (NerualNetwork * NN)
{
    uint16_t hiddenNeuronCount = 0;
    printf("Input Layer Weights:\n\n");
    vPrintMatrix(NN->inputLayer->inputLayer);
    printf("------------------------------------------------------\n");
    printf("\nHidden Layer Weights & Biases:\n\n");
    for (uint16_t currHL = 0; currHL < NN->numHiddenLayers; currHL++)
    {
        printf("Hidden Layer %d Weights:\n", currHL);
        vPrintMatrix(NN->hiddenLayers[currHL]->hiddenLayer);
        printf("\nHidden Layer %d Biases:\n", currHL);
        vPrintMatrix(NN->hiddenLayers[currHL]->biases);
        printf("\n\n");
        hiddenNeuronCount += NN->hiddenLayers[currHL]->hiddenLayer->cols;
    }
    printf("------------------------------------------------------\n");
    printf("Output Layer Weights & Biases:\n\nOutput Layer Weights:\n");
    vPrintMatrix(NN->outputLayer->outputLayer);
    printf("\nOutput Layer Biases:\n");
    vPrintMatrix(NN->outputLayer->biases);
    printf("\nSummary:\n");
    printf("Number of Inputs: %d\n", NN->inputLayer->inputLayer->cols);
    printf("Number of Outputs: %d\n", NN->outputLayer->outputLayer->cols);
    printf("Number of Neurons: %d\n", NN->inputLayer->inputLayer->cols + hiddenNeuronCount + NN->outputLayer->outputLayer->cols);
    printf("Number of Hidden Neurons: %d\n", hiddenNeuronCount);
    //printf("Total Number of Connections: %d\n", hiddenNeuronCount*(NN->numHiddenLayers+1));

}