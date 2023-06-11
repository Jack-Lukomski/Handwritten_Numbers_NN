#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include <armadillo>

namespace ActivationFunctions {
    arma::mat sigmoid(const arma::mat& matrix);
    arma::mat tanh_act(const arma::mat& matrix);
    arma::mat relu(const arma::mat& matrix);
}

#endif /* ACTIVATION_FUNCTIONS_HPP */