#include "neuralNetwork.h"

static Layer * xCreateInputLayer(uint16_t numInputs)
{
    Layer * newInputLayer = (Layer *) malloc(sizeof(Layer));
    newInputLayer->numNeurons = numInputs;
    newInputLayer->neurons = (Neuron *) malloc(numInputs * sizeof(Neuron));

    for (uint16_t currNuron = 0; currNuron < numInputs; currNuron++)
    {
        newInputLayer->neurons[currNuron].numInputs = 0;
        newInputLayer->neurons[currNuron].weights = NULL;
        newInputLayer->neurons[currNuron].bias = 0.0;
    }

    return newInputLayer;
}

static Layer * xCreateOutputLayer(uint16_t numOutputs, uint16_t numInputs, double ** weights, double * biases)
{
    Layer * newOutputLayer = (Layer *) malloc(sizeof(Layer));
    newOutputLayer->numNeurons = numOutputs;
    newOutputLayer->neurons = (Neuron *) malloc(numOutputs * sizeof(Neuron));

    for (uint16_t currNeuron = 0; currNeuron < numOutputs; currNeuron++)
    {
        newOutputLayer->neurons[currNeuron].numInputs = numInputs;
        newOutputLayer->neurons[currNeuron].weights = (double *) malloc(numInputs * sizeof(double));
        for (uint16_t currWeight = 0; currWeight < numInputs; currWeight++)
        {
            newOutputLayer->neurons[currNeuron].weights[currWeight] = weights[currNeuron][currWeight];
        }
        newOutputLayer->neurons[currNeuron].bias = biases[currNeuron];
    }
    return newOutputLayer;
}

/* TODO: Seg falt from this function, something with memory management*/
static Layer * xCreateHiddenLayers(uint16_t numHiddenLayers, uint16_t * numHiddenNeurons, uint16_t numInputs, double *** weights, double ** biases)
{
    Layer * hiddenLayers = (Layer *) malloc(numHiddenLayers * sizeof(Layer));

    for (uint16_t currLayer = 0; currLayer < numHiddenLayers; currLayer++)
    {
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
        free(currHiddenLayer);
    }
    return hiddenLayers;
}



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
}