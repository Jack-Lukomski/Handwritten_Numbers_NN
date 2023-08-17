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
    NeuralNetwork(NeuralNetArch_t & architecture, ActivationType af);

    void forwardProp();
    void learn(NeuralNetwork gradient, float learnRate);
    NeuralNetwork getGradient_fd(const std::vector<arma::mat> & inputs, const std::vector<arma::mat> & outputs, float eps);
    void backprop(const std::vector<arma::mat> & inputs, const std::vector<arma::mat> & outputs, float learnRate, unsigned int numEpochs);
    float getCost(const std::vector<arma::mat> & inputs, const std::vector<arma::mat> & outputs);
    arma::mat getOutput();
    void setInput(const arma::mat & input);
    void randomize(float min, float max);
    void print() const;

private:
    size_t _layerCount;
    std::vector<arma::mat> _weights;
    std::vector<arma::mat> _biases;
    std::vector<arma::mat> _activations;
    ActivationType _af;
    NeuralNetArch_t _arch;
};

#endif /* NEURALNETWORK_HPP */