#include <iostream>
#include <vector>
#include "../include/NeuralNetwork.hpp"

int main () 
{
    std::vector<uint32_t> hiddenN = {2, 2};
    NeuralNetwork nn(2, 2, hiddenN, 2);

    //arma::mat input = arma::zeros<arma::mat>(2,1);
    arma::mat input(1,2);
    input << 1.0 << 2.0 << arma::endr;



    nn.randomize();
    arma::mat output = nn.forwardProp(input);
    nn.printNetwork();

    std::cout << "Output: " << output << std::endl;

    return 0;    
}