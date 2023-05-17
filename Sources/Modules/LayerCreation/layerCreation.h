#ifndef LAYER_CREATION_H
#define LAYER_CREATION_H

#include "../Matrix/matrix.h"
#include "../NeuralNetwork/neuralNetwork.h"
#include <stdio.h>
#include <stdbool.h>

#define NUM_HIDDEN_LAYERS 2

Matrix * INPUT_LAYER_WEIGHTS;

Matrix * HIDDEN_LAYER_WEIGHTS[NUM_HIDDEN_LAYERS];
Matrix * HIDDEN_LAYER_BIASES[NUM_HIDDEN_LAYERS];

Matrix * OUTPUT_LAYER_WEIGHTS;
Matrix * OUTPUT_LAYER_BIASES;

typedef struct {
    FILE * InputWeights;
    FILE ** HiddenLayerWeights;
    FILE ** HiddenBiases;
    FILE * OutputWeights;
    FILE * OutputBiases;
} LayerCSVFiles;

typedef struct {
    uint16_t rows;
    uint16_t cols;
} fileStructure;


void vCreateNNLayerMatricies (void);
static void vCreateInputLayerWeights(FILE * fp);

static LayerCSVFiles * xConstructLayerCSVFiles (void);
static fileStructure xGetFileStructure (FILE * f);
static double * xGetFileData (FILE * f, fileStructure currFileStructure);

void vUpdateInputLayerWeights (Matrix * inputLayerWeights, NerualNetwork * NN);
void vUpdateOutputLayerWeights (Matrix * outputLayerWeights, NerualNetwork * NN);
void vUpdateOutputLayerBiases (Matrix * outputLayerBiases, NerualNetwork * NN);
void vUpdateHiddenLayerWeights (Matrix * hiddenLayerWeights, NerualNetwork * NN, uint16_t hiddenLayerNum);
void vUpdateHiddenLayerBiases (Matrix * hiddenLayerBiases, NerualNetwork * NN, uint16_t hiddenLayerNum);

#endif