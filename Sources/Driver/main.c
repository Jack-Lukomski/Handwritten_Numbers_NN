#include "../Modules/Matrix/matrix.h"
#include "../Modules/NeuralNetwork/neuralNetwork.h"
#include "../Modules/ActivationFunctions/activationFunctions.h"
#include "../Modules/LayerCreation/layerCreation.h"
#include <stdio.h>

extern Matrix * INPUT_LAYER_WEIGHTS;
extern Matrix * HIDDEN_LAYER_WEIGHTS[NUM_HIDDEN_LAYERS];
extern Matrix * HIDDEN_LAYER_BIASES[NUM_HIDDEN_LAYERS];
extern Matrix * OUTPUT_LAYER_WEIGHTS;
extern Matrix * OUTPUT_LAYER_BIASES;

int main (void)
{
    vCreateNNLayerMatricies();

    NerualNetwork * NN = xConstructNeuralNetwork(INPUT_LAYER_WEIGHTS, NUM_HIDDEN_LAYERS, HIDDEN_LAYER_WEIGHTS, HIDDEN_LAYER_BIASES, OUTPUT_LAYER_WEIGHTS, OUTPUT_LAYER_BIASES);
    vPrintAllLayers(NN);

    return 0;
}

