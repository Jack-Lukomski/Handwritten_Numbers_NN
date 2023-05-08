#include "../Modules/Matrix/matrix.h"
#include "../Modules/NeuralNetwork/neuralNetwork.h"
#include "../Modules/ActivationFunctions/activationFunctions.h"
#include "../Modules/LayerCreation/layerCreation.h"
#include <stdio.h>

// double InputLayerArr[2] = {
//     1.0, 1.0,
// };

// double HiddenLayerArr[8] = {
//     2.0, 2.0, 2.0, 2.0,
//     3.0, 3.0, 3.0, 3.0,
// };

// double HiddenLayerBiases[4] = {
//     4.0, 4.0, 4.0, 4.0,
// };

// double OutputLayerArr[8] = {
//     5.0, 5.0,
//     5.0, 5.0,
//     5.0, 5.0,
//     5.0, 5.0,
// };

// double OutputLayerBiases[2] = {
//     6.0, 6.0,
// };

// int main(void)
// {
//     Matrix * INPUT_LAYER = xCreateMatrix(1, 2, InputLayerArr);

//     Matrix * HIDDEN_LAYER[1];
//     Matrix * HIDDEN_LAYER_BIASES[1];

//     HIDDEN_LAYER[0] =  xCreateMatrix(2, 4, HiddenLayerArr);
//     HIDDEN_LAYER_BIASES[0] = xCreateMatrix(1, 4, HiddenLayerBiases);

//     Matrix * OUTPUT_LAYER = xCreateMatrix(4, 2, OutputLayerArr);
//     Matrix * OUTPUT_LAYER_BIASES = xCreateMatrix(1, 2, OutputLayerBiases);

//     NerualNetwork * NN = xConstructNeuralNetwork(INPUT_LAYER, 1, HIDDEN_LAYER, HIDDEN_LAYER_BIASES, OUTPUT_LAYER, OUTPUT_LAYER_BIASES);
//     // Matrix * m = xComputeOutputSums(NN, Sigmoid);
//     // vPrintMatrix(m);
//     vPrintAllLayers(NN);
// }

int main (void)
{
    vCreateNNLayerMatricies();

    return 0;
}

