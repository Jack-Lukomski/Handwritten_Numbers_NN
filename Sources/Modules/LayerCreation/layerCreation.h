#ifndef LAYER_CREATION_H
#define LAYER_CREATION_H

#include "../Matrix/matrix.h"
#include <stdio.h>
#include <stdbool.h>

#define NUM_HIDDEN_LAYERS 1

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
void vUpdateInputLayerWeights (Matrix * inputLayerWeights);
void vUpdateOutputLayerWeights (Matrix * outputLayerWeights);
void vUpdateOutputLayerBiases (Matrix * outputLayerBiases);
void vUpdateHiddenLayerWeights (Matrix * hiddenLayerWeights, uint16_t hiddenLayerNum);
void vUpdateHiddenLayerBiases (Matrix * hiddenLayerBiases, uint16_t hiddenLayerNum);

#endif