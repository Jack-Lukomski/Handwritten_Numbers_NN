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
    std::vector<arma::mat> xorTrainInputData;
    std::vector<arma::mat> xorTrainOutputData;

    arma::mat zz_input = {{0.0, 0.0}};
    arma::mat zz_output = arma::mat(1, 1).fill(0.0);
    xorTrainInputData.push_back(zz_input);
    xorTrainOutputData.push_back(zz_output);

    arma::mat zo_input = {{0.0, 1.0}};
    arma::mat zo_output = arma::mat(1, 1).fill(1.0);
    xorTrainInputData.push_back(zo_input);
    xorTrainOutputData.push_back(zo_output);

    arma::mat oz_input = {{1.0, 0.0}};
    arma::mat oz_output = arma::mat(1, 1).fill(1.0);
    xorTrainInputData.push_back(oz_input);
    xorTrainOutputData.push_back(oz_output);

    arma::mat oo_input = {{1.0, 1.0}};
    arma::mat oo_output = arma::mat(1, 1).fill(0.0);
    xorTrainInputData.push_back(oo_input);
    xorTrainOutputData.push_back(oo_output);

    std::vector<uint32_t> numHiddedenNeurons = {4};
    NeuralNetwork xorTest(2, 1, numHiddedenNeurons, 1);

    xorTest.randomize();
    xorTest.train(xorTrainInputData, xorTrainOutputData, 0.3, 10000, ActivationFunctions::sigmoid, ActivationFunctions::sigmoidDerivative);

    std::vector<arma::mat> outputs = xorTest.forwardProp(xorTrainInputData[1], ActivationFunctions::sigmoid);

    std::cout << outputs[1] << std::endl;

    return 0;
}