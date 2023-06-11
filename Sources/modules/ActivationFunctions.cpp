#include "../include/ActivationFunctions.hpp"

namespace ActivationFunctions {
    arma::mat sigmoid(const arma::mat& matrix) 
    {
        return 1.0 / (1.0 + arma::exp(-matrix));
    }

    arma::mat tanh_act(const arma::mat& matrix) 
    {
        return arma::tanh(matrix);
    }

    arma::mat relu(const arma::mat& matrix) 
    {
        return arma::clamp(matrix, 0.0, arma::datum::inf);
    }
}
