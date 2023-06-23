#include "../include/A_Func.hpp"
#include <cmath>

A_Func::A_Func(ActivationType type) {
    functions = {
        {ActivationType::SIGMOID, [](arma::mat& x) { x = 1 / (1 + arma::exp(-x)); }},
        {ActivationType::RELU, [](arma::mat& x) { x.transform( [](double val) { return val > 0 ? val : 0; } ); }},
        {ActivationType::TANH, [](arma::mat& x) { x = arma::tanh(x); }},
    };

    currentFunction = functions[type];
}

void A_Func::apply(arma::mat& matrix) {
    currentFunction(matrix);
}
