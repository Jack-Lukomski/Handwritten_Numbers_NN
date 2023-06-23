#include <iostream>
#include <vector>
#include "../include/NeuralNetwork.hpp"
#include "../include/A_Func.hpp"

NeuralNetArch_t arch = {2, 4, 1};
arma::mat a = {{1.32, 4.55}};

int main ()
{
    NeuralNetwork nn(arch);
    nn.print();
    nn.randomize(0, 10);
    nn.print();
    nn.setInput(a);
    nn.forwardProp(A_Func_Type::Sigmoid);
    std::cout << "Output: " << nn.getOutput() << std::endl;
    return 0;
}