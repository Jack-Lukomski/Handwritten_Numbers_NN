#include <iostream>
#include <vector>
#include "../include/NeuralNetwork.hpp"
#include "../include/A_Func.hpp"

NeuralNetArch_t arch = {2, 2, 2, 1};

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
    nn.randomize(0, 1);
    nn.print();
    // NeuralNetwork gradient = nn.backprop(inputs, outputs);
    nn.backprop(inputs, outputs, 0.1);
    std::cout << "\n\n\n\n";
    nn.print();
    // for (size_t i = 0; i < 100000; ++i) {
    //     NeuralNetwork gradient = nn.getGradient_fd(inputs, outputs, 1e-1);
    //     nn.learn(gradient, 1e-1);
    //     //std::cout << nn.getCost(inputs, outputs) << std::endl;
    // }

    // for (size_t i = 0; i < inputs.size(); ++i) {
    //     std::cout << "Input is: " << inputs[i] << "Output should be: " << outputs[i];
    //     nn.setInput(inputs[i]);
    //     nn.forwardProp();
    //     std::cout << "The output is: " << nn.getOutput() << "\n\n" << std::endl;
    // }

    return 0;
}