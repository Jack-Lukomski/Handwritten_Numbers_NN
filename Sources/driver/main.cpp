#include <iostream>
#include <vector>
#include "../include/NeuralNetwork.hpp"
#include "../include/A_Func.hpp"

NeuralNetArch_t arch = {2, 4, 1};

int main ()
{
    NeuralNetwork nn(arch);
    nn.print();
    nn.randomize(0, 10);
    nn.print();
    nn.forwardProp(A_Func_Type::Sigmoid);
    std::cout << "Output: " << nn.getOutput() << std::endl;
    return 0;
}