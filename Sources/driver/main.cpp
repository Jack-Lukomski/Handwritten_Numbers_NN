#include <iostream>
#include <vector>
#include "../include/NeuralNetwork.hpp"

int main () 
{
    std::vector<uint32_t> hiddenN = {4, 4};
    NeuralNetwork nn(2, 2, hiddenN, 2);

    nn.printNetwork();
    return 0;    
}