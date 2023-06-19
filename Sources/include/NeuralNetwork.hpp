#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "ActivationFunctions.hpp"
#include <vector>
#include <functional>
#include <armadillo>

typedef std::vector<uint32_t> NeuralNetArch_t;

class NeuralNetwork {
public:
    NeuralNetwork(NeuralNetArch_t & architecture);

    void randomize(float min, float max);
    void print() const;

private:
    size_t _layerCount;
    std::vector<arma::mat> _weights;
    std::vector<arma::mat> _biases;
    std::vector<arma::mat> _activations;
};

#endif /* NEURALNETWORK_HPP */