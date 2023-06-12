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

    arma::mat sigmoidDerivative(const arma::mat& output)
    {
        return output % (1 - output);
    }

    arma::mat tanhDerivative(const arma::mat& output)
    {
        return 1 - arma::pow(output, 2);
    }

    arma::mat reluDerivative(const arma::mat& input)
    {
        return arma::conv_to<arma::mat>::from(input > 0);
    }
}
