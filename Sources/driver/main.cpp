#include <iostream>
#include <vector>
#include "../include/NeuralNetwork.hpp"
#include "../include/A_Func.hpp"

NeuralNetArch_t arch = {2, 4, 1};

arma::mat a = {{0, 0}};
arma::mat b = {{1, 0}};
arma::mat c = {{0, 1}};
arma::mat d = {{1, 1}};

arma::mat ao = arma::mat(1, 1, arma::fill::zeros);
arma::mat bo = arma::mat(1, 1, arma::fill::ones);
arma::mat co = arma::mat(1, 1, arma::fill::ones);
arma::mat doo = arma::mat(1, 1, arma::fill::zeros);

std::vector<arma::mat> inputs = {a, b, c, d};
std::vector<arma::mat> outputs = {ao, bo, co, doo};

int main ()
{
    NeuralNetwork nn(arch, ActivationType::SIGMOID);
    nn.print();
    nn.randomize(0, 1);
    nn.print();
    nn.setInput(a);
    nn.forwardProp();
    std::cout << "Output: " << nn.getOutput() << std::endl;
    std::cout << "cost: " << nn.getCost(inputs, outputs) << std::endl;
    return 0;
}