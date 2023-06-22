#ifndef A_FUNC_HPP
#define A_FUNC_HPP

#include <armadillo>
#include <iostream>
#include <vector>

enum class A_FuncType {
    Sigmoid,
    Tahn,
    Relu,
};

class A_Func {
public:
    A_Func(A_FuncType type);

    arma::mat apply(arma::mat& matrix);
    arma::mat applyPrime(arma::mat &matrix);
private:
    std::function<void(arma::mat&)> _af;
    std::function<void(arma::mat&)> _afd;

    void sigmoid(arma::mat& matrix);
    void tanh(arma::mat& matrix);
    void relu(arma::mat& matrix);
    void sigmoidPrime(arma::mat& output);
    void tanhPrime(arma::mat& output);
    void reluPrime(arma::mat& input);
};

#endif /* A_FUNC_HPP */