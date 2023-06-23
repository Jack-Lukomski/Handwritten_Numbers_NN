#ifndef A_FUNC_HPP
#define A_FUNC_HPP

#include <map>
#include <functional>
#include <armadillo>

enum class ActivationType {
    SIGMOID,
    RELU,
    TANH,
};

class A_Func {
    private:
        std::map<ActivationType, std::function<void(arma::mat&)>> functions;

        std::function<void(arma::mat&)> currentFunction;

    public:
        A_Func(ActivationType type);

        void apply(arma::mat& matrix);
};

#endif // A_FUNC_HPP
