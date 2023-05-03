#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    uint16_t numInputs;
    double * weights;
    double bias;
} Neuron;

typedef struct
{
    uint16_t numNeurons;
    Neuron * neurons;
} Layer;

typedef struct {
    uint16_t numInputs;
    Layer inputLayer;
    uint16_t numHiddenLayers;
    Layer * hiddenLayers;
    uint16_t numOutputs;
    Layer outputLayer;
} NeuralNetwork;

static Layer * xCreateInputLayer(uint16_t numInputs);
static Layer * xCreateOutputLayer(uint16_t numOutputs, uint16_t numInputs, double ** weights, double * biases);
static Layer * xCreateHiddenLayers(uint16_t numHiddenLayers, uint16_t * numHiddenNeurons, uint16_t numInputs, double *** weights, double ** biases);
NeuralNetwork * xCreateNeuralNetwork(uint16_t numInputs, uint16_t numHiddenLayers, uint16_t * numHiddenNurons, uint16_t numOutputs, double *** hiddenWeights, double ** hiddenBiases, double ** outputWeights, double * outputBiases);

#endif