#include <iostream>
#include <vector>
#include "../include/NeuralNetwork.hpp"
#include "../include/ActivationFunctions.hpp"

typedef struct {
    arma::mat input;
    arma::mat output;
} XorTrainData;

int main () 
{
    XorTrainData zz;
    XorTrainData zo;
    XorTrainData oz;
    XorTrainData oo;

    zz.input = arma::zeros<arma::mat>(1, 2);
    zz.output = arma::zeros<arma::mat>(1, 1);
    zo.input = arma::zeros<arma::mat>(1, 2);
    zo.output = arma::zeros<arma::mat>(1, 1);
    oz.input = arma::zeros<arma::mat>(1, 2);
    oz.output = arma::zeros<arma::mat>(1, 1);
    oo.input = arma::zeros<arma::mat>(1, 2);
    oo.output = arma::zeros<arma::mat>(1, 1);

    zz.input << 0 << 0 << arma::endr;
    zz.output << 0 << arma::endr;

    zo.input << 0 << 1 << arma::endr;
    zo.output << 1 << arma::endr;

    oz.input << 1 << 0 << arma::endr;
    oz.output << 1 << arma::endr;

    oo.input << 1 << 1 << arma::endr;
    oo.output << 0 << arma::endr;

    std::vector<uint32_t> hiddenLayerNeurons = {4};
    NeuralNetwork xorModel(2, 1, hiddenLayerNeurons, 1);
    xorModel.randomize();

    xorModel.train(zz.input, zz.output, 0.05, 1, ActivationFunctions::sigmoid, ActivationFunctions::sigmoidDerivative);

    // std::cout << "Outputs:" << std::endl; 
    // for (const auto & curr: result)
    // {
    //     std::cout << curr << std::endl;
    // }

    return 0;
}