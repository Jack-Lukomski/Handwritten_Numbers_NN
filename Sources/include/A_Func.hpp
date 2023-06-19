#ifndef A_FUNC_HPP
#define A_FUNC_HPP

#include <armadillo>

enum class A_FuncType {
    Sigmoid,
    Tahn,
    Relu,
};

class A_Func {
public:
    A_Func(A_FuncType type);
private:
    arma::mat (*_af)(const arma::mat &);
    arma::mat (*_afd)(const arma::mat &);

    arma::mat sigmoid(const arma::mat& matrix);
    arma::mat tanh_act(const arma::mat& matrix);
    arma::mat relu(const arma::mat& matrix);
    arma::mat sigmoidDerivative(const arma::mat& output);
    arma::mat tanhDerivative(const arma::mat& output);
    arma::mat reluDerivative(const arma::mat& input);
};

#endif /* A_FUNC_HPP */