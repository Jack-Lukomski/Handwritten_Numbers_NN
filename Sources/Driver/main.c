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
    double inputMatrix[2] = {
        1.2, 2.2
    };

    double firstHL[8] = {
        1.2, 3.4, 5.5, 5.6, // connection between one neuron
        4.5, 6.7, 3.2, 6.1,
    };

    double secondHL[8] = {
        1.9, 3.4, 5.5, 5.6, // connection between one neuron
        4.5, 6.7, 3.2, 1.1,
    };

    double firstHLBiases[4] = {
        1.2, 3.4, 3.2, 4.4,
    };

    double secondHLBiases[4] = {
        8.2, 3.2, 3.2, 5.4,
    };

    double oMatrix[2] = {
        1.2, 2.2
    };

    double oMatrixBiases[2] = {
        1.2, 2.2
    };

    Matrix * ipMatrix = xCreateMatrix(1, 2, inputMatrix);

    Matrix *hlArr[2];
    Matrix *hlBias[2];

    hlArr[0] = xCreateMatrix(2, 4, firstHL);
    hlBias[0] = xCreateMatrix(1, 4, firstHLBiases);

    hlArr[1] = xCreateMatrix(2, 4, secondHL);
    hlBias[1] = xCreateMatrix(1, 4, secondHLBiases);

    Matrix * op = xCreateMatrix(1, 2, oMatrix);
    Matrix * opB = xCreateMatrix(1, 2, oMatrixBiases);

    NerualNetwork * NN = xConstructNeuralNetwork(ipMatrix, 2, hlArr, hlBias, op, opB);

    // vPrintMatrix(newIL->inputLayer);
    // printf("\n\n");
    vPrintMatrix(NN->hiddenLayers[1]->hiddenLayer);
    printf("\n\n");
    //printf("\n\n");
    //vPrintMatrix(hlMatric);
}

