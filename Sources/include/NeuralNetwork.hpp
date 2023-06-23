#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "A_Func.hpp"
#include <vector>
#include <functional>
#include <armadillo>
#include <cassert>

typedef std::vector<uint32_t> NeuralNetArch_t;

class NeuralNetwork {
public:
    NeuralNetwork(NeuralNetArch_t & architecture);

    void forwardProp(ActivationType af);
    arma::mat getOutput();
    void setInput(arma::mat & input);
    void randomize(float min, float max);
    void print() const;

private:
    size_t _layerCount;
    std::vector<arma::mat> _weights;
    std::vector<arma::mat> _biases;
    std::vector<arma::mat> _activations;
    NeuralNetArch_t _arch;
};

#endif /* NEURALNETWORK_HPP */