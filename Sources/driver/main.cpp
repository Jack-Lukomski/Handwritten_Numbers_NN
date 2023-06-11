#include <iostream>
#include <vector>
#include "../include/NeuralNetwork.hpp"

int main () 
{
    std::vector<uint32_t> hiddenN = {2, 2};
    NeuralNetwork nn(2, 2, hiddenN, 2);

    nn.randomize();
    nn.printNetwork();
    nn.forwardProp();
    nn.printNetwork();
    return 0;    
}