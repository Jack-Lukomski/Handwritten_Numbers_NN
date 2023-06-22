#include "../include/A_Func.hpp"

A_Func::A_Func(A_FuncType type) 
{
    switch (type)
    {
    case A_FuncType::Sigmoid:
        _af = [this](arma::mat& matrix){this->sigmoid(matrix);};
        _afd = [this](arma::mat& output){this->sigmoidPrime(output);};
        break;
    case A_FuncType::Relu:
        _af = [this](arma::mat& matrix){this->relu(matrix);};
        _afd = [this](arma::mat& input){this->reluPrime(input);};
        break;
    case A_FuncType::Tahn:
        _af = [this](arma::mat& matrix){this->tanh(matrix);};
        _afd = [this](arma::mat& output){this->tanhPrime(output);};
        break;
    default:
        break;
    }
}

void A_Func::sigmoid(arma::mat& matrix) 
{
    matrix = 1.0 / (1.0 + arma::exp(-matrix));
}

void A_Func::tanh(arma::mat& matrix) 
{
    matrix = arma::tanh(matrix);
}

void A_Func::relu(arma::mat& matrix) 
{
    matrix = arma::clamp(matrix, 0.0, arma::datum::inf);
}

// may need to return a new matrix, FIX!!
void A_Func::sigmoidPrime(arma::mat& output)
{
    output = output % (1 - output);
}

void A_Func::tanhPrime(arma::mat& output)
{
    output = 1 - arma::pow(output, 2);
}

void A_Func::reluPrime(arma::mat& input)
{
    input = arma::conv_to<arma::mat>::from(input > 0);
}