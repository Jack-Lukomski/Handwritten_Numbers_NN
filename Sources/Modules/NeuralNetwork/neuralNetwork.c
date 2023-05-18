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

Matrix * xForwardPropagation (NerualNetwork * NN, e_FunctionOption activationFunction)
{
    ActivationFunction * currActivationFunction = xCreateActivationFunction(activationFunction);
    Matrix * outputSumMatrix = (Matrix *) malloc(sizeof(Matrix));

    for (uint16_t currLayer = 0; currLayer < NN->numHiddenLayers; currLayer++)
    {
        if (currLayer == 0)
        {
            outputSumMatrix = xMatrixAdd(xDotProduct(NN->inputLayer->inputLayer, NN->hiddenLayers[currLayer]->hiddenLayer), NN->hiddenLayers[currLayer]->biases);
            vApplyActivationFunction(outputSumMatrix, currActivationFunction);
        }
        else 
        {
            outputSumMatrix = xMatrixAdd(xDotProduct(outputSumMatrix, NN->hiddenLayers[currLayer]->hiddenLayer), NN->hiddenLayers[currLayer]->biases);
            vApplyActivationFunction(outputSumMatrix, currActivationFunction);
        }
    }

    outputSumMatrix = xMatrixAdd(xDotProduct(outputSumMatrix, NN->outputLayer->outputLayer), NN->outputLayer->biases);
    vApplyActivationFunction(outputSumMatrix, currActivationFunction);

    free(currActivationFunction);

    return outputSumMatrix;
}

void vTrainNeuralNetwork (NerualNetwork * NN, Matrix * expectedOutputMatrix, uint16_t numEpochs, double learningRate)
{
    for (uint16_t currEpoch = 0; currEpoch < numEpochs; currEpoch++)
    {
        Matrix * outputMatrix = xForwardPropagation(NN, Sigmoid);
        Matrix * outputError = xMatrixSubtract(outputError, expectedOutputMatrix);
        outputError = xMatrixSquareEachElement(outputError);
    }
}

static void vApplyActivationFunction(Matrix * m, ActivationFunction * function)
{
    for (uint16_t currMatrixVal = 0; currMatrixVal < m->cols*m->rows; currMatrixVal++)
    {
        m->matrixData[currMatrixVal] = function->function(m->matrixData[currMatrixVal]);
    }
}

void vReconstructInputLayer(NerualNetwork * NN, Matrix * newData)
{
    free(NN->inputLayer);
    NN->inputLayer = (InputLayer *) malloc(sizeof(InputLayer));
    NN->inputLayer->inputLayer = newData;
}

void vReconstructOutputWeights(NerualNetwork * NN, Matrix * newData)
{
    free(NN->outputLayer->outputLayer);
    NN->outputLayer->outputLayer = (Matrix *) malloc(sizeof(Matrix));
    NN->outputLayer->outputLayer = newData;
}

void vReconstructOutputBiases(NerualNetwork * NN, Matrix * newData)
{
    free(NN->outputLayer->biases);
    NN->outputLayer->biases = (Matrix *) malloc(sizeof(Matrix));
    NN->outputLayer->biases = newData;
}

void vReconstructHiddenWeights(NerualNetwork * NN, Matrix * newData, uint16_t currHiddenLayer)
{
    free(NN->hiddenLayers[currHiddenLayer]->hiddenLayer);
    NN->hiddenLayers[currHiddenLayer]->hiddenLayer = (Matrix *) malloc(sizeof(Matrix));
    NN->hiddenLayers[currHiddenLayer]->hiddenLayer = newData;
}

void vReconstructHiddenBiases(NerualNetwork * NN, Matrix * newData, uint16_t currHiddenLayer)
{
    free(NN->hiddenLayers[currHiddenLayer]->biases);
    NN->hiddenLayers[currHiddenLayer]->biases = (Matrix *) malloc(sizeof(Matrix));
    NN->hiddenLayers[currHiddenLayer]->biases = newData;
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
}