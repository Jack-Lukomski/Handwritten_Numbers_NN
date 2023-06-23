#include "../include/A_Func.hpp"

A_Func::A_Func(A_Func_Type func) {
    switch(func) {
        case A_Func_Type::Sigmoid:
            activationFunction = [](arma::mat& m) {
                m = 1.0 / (1.0 + arma::exp(-m));
            };
            derivativeFunction = [](arma::mat& m) {
                arma::mat sigmoid = 1.0 / (1.0 + arma::exp(-m));
                m = sigmoid % (1 - sigmoid);  // element-wise multiplication
            };
            break;
        case A_Func_Type::Tanh:
            activationFunction = [](arma::mat& m) {
                m = arma::tanh(m);
            };
            derivativeFunction = [](arma::mat& m) {
                m = 1.0 - arma::pow(arma::tanh(m), 2);
            };
            break;
        case A_Func_Type::ReLU:
            activationFunction = [](arma::mat& m) {
                m.transform([](double val) { return val < 0.0 ? 0.0 : val; }); // element-wise ReLU
            };
            derivativeFunction = [](arma::mat& m) {
                m.transform([](double val) { return val < 0.0 ? 0.0 : 1.0; }); // element-wise derivative of ReLU
            };
            break;
        default:
            throw std::invalid_argument("Invalid activation function type.");
    }
}

void A_Func::apply(arma::mat& m) {
    activationFunction(m);
}

void A_Func::applyPrime(arma::mat& m) {
    derivativeFunction(m);
}
