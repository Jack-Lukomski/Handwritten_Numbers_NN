#include <iostream>
#include <vector>
#include "../include/NeuralNetwork.hpp"
#include "../include/A_Func.hpp"

NeuralNetArch_t arch = {2, 4, 1};
arma::mat a = {{1, 0}};

int main ()
{
    NeuralNetwork nn(arch);
    nn.print();
    nn.randomize(0, 1);
    nn.print();
    nn.setInput(a);
    nn.forwardProp(ActivationType::SIGMOID);
    std::cout << "Output: " << nn.getOutput() << std::endl;
    return 0;
}