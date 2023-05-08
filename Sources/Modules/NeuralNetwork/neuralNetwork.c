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
<<<<<<< HEAD
        Layer * currHiddenLayer = (Layer *) malloc(sizeof(Layer));
        currHiddenLayer->numNeurons = numHiddenNeurons[currLayer];
        currHiddenLayer->neurons = (Neuron *) malloc(numHiddenNeurons[currLayer] * sizeof(Neuron));
        for (uint16_t currNeuron = 0; currNeuron < numHiddenNeurons[currLayer]; currNeuron++)
        {
            currHiddenLayer->neurons[currNeuron].numInputs = numInputs;
            currHiddenLayer->neurons[currNeuron].weights = (double *) malloc(numInputs * sizeof(double));
            for (uint16_t currWeight = 0; currWeight < numInputs; currWeight++)
            {
                currHiddenLayer->neurons[currNeuron].weights[currWeight] = weights[currLayer][currNeuron][currWeight];
            }
            currHiddenLayer->neurons[currNeuron].bias = biases[currLayer][currNeuron];
        }
        hiddenLayers[currLayer] = *currHiddenLayer;
        //free(currHiddenLayer);
=======
        newNN->hiddenLayers[currHL] = xConstructHiddenLayer(hiddenLayerMatricies[currHL], hiddenLayerBiases[currHL]);
>>>>>>> RedoingNNModuleToIncoperateMatrix
    }

<<<<<<< HEAD
NeuralNetwork * xCreateNeuralNetwork(uint16_t numInputs, uint16_t numHiddenLayers, uint16_t * numHiddenNurons, uint16_t numOutputs, double *** hiddenWeights, double ** hiddenBiases, double ** outputWeights, double * outputBiases)
{
    Layer * inputLayer = xCreateInputLayer(numInputs);
    Layer * hiddenLayers = xCreateHiddenLayers(numHiddenLayers, numHiddenNurons, numInputs, hiddenWeights, hiddenBiases);
    Layer * outputLayer = xCreateOutputLayer(numOutputs, numHiddenNurons[numHiddenLayers-1], outputWeights, outputBiases);

    NeuralNetwork * neuralNetwork = (NeuralNetwork *) malloc(sizeof(NeuralNetwork));
    neuralNetwork->numInputs = numInputs;
    neuralNetwork->numHiddenLayers = numHiddenLayers;
    neuralNetwork->numOutputs = numOutputs;
    neuralNetwork->inputLayer = *inputLayer;
    neuralNetwork->hiddenLayers = hiddenLayers;
    neuralNetwork->outputLayer = *outputLayer;

    free(inputLayer);
    free(outputLayer);

    return neuralNetwork;
=======
    return newNN;
>>>>>>> RedoingNNModuleToIncoperateMatrix
}