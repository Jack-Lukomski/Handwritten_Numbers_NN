#include <iostream>
#include <vector>
#include "../include/NeuralNetwork.hpp"
#include "../include/ActivationFunctions.hpp"

int main () 
{
    std::vector<uint32_t> hiddenN = {4};
    NeuralNetwork nn(2, 1, hiddenN, 2);

    arma::mat input(1,2);
    input << 1.0 << 2.0 << arma::endr;



    nn.randomize();
    arma::mat output = nn.forwardProp(input, ActivationFunctions::sigmoid);
    nn.printNetwork();

    std::cout << "Output: " << output << std::endl;

    return 0;
}