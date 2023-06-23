#ifndef A_FUNC_HPP
#define A_FUNC_HPP

#include <armadillo>
#include <functional>

enum class A_Func_Type { Sigmoid, Tanh, ReLU };

class A_Func {
public:
    A_Func(A_Func_Type func);
    void apply(arma::mat& m);
    void applyPrime(arma::mat& m);

private:
    std::function<void(arma::mat&)> activationFunction;
    std::function<void(arma::mat&)> derivativeFunction;
};

#endif // A_FUNC_HPP
