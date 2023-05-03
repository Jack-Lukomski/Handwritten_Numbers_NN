#include "../Modules/Matrix/matrix.h"
#include "../Modules/NeuralNetwork/neuralNetwork.h"
#include <stdio.h>
// int main(void)
// {
//     double m1[9] = {1.2, 34.2, 
//                     21.3, 3.4};
                      
//     double m2[9] = {1.6, 3.2, 
//                     21.3, 4.5};

//     double m3[9] = {1.4, 1.1, 5.4, 
//                     4.5, 6.5, 6.7,
//                     7.7, 9.0, 34.2};

//     double m4[3] = {1.2, 4.4, 44.3};

//     Matrix * ma1 = xCreateMatrix(2, 2, m1);
//     Matrix * ma2 = xCreateMatrix(2, 2, m2);
//     Matrix * ma3 = xCreateMatrix(3, 3, m3);
//     Matrix * ma4 = xCreateMatrix(1, 3, m4);

//     Matrix * dotp = xDotProduct(ma1, ma2);

//     vPrintMatrix(dotp);

//     return 0;
// }

int main(void)
{
    uint16_t numInputs = 2;
    uint16_t numHiddenLayers = 2;
    uint16_t numHiddenNeurons[2] = {2, 2};
    uint16_t numOutputs = 1;

    double hiddenWeights[2][2][2] = {
        {
            {1.0, 2.0},
            {3.0, 4.0}
        },
        {
            {5.0, 6.0},
            {7.0, 8.0}
        }
    };

    double hiddenBiases[2][2] = {
        {3.2, 1.2},
        {1.6, 8.6}
    };

    double outputWeights[1][2] = {
        {2.8, 8.8}
    };

    double outputBiases[1] = {1.2};

    // Create temporary arrays for weights and biases
    double ** hiddenWeights2D = (double **) hiddenWeights;
    double * hiddenBiases1D = (double *) hiddenBiases;

    // Create the neural network
    NeuralNetwork * NN = xCreateNeuralNetwork(numInputs, numHiddenLayers, numHiddenNeurons, numOutputs, &hiddenWeights2D, &hiddenBiases1D, (double **) outputWeights, outputBiases);

    printf("s");

    free(NN);
    return 0;
}

