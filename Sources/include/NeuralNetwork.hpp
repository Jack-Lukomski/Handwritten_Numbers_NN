#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "Layer.hpp"
#include "ActivationFunctions.hpp"
#include <vector>
#include <functional>
#include <armadillo>
#include <random>

class NeuralNetwork {
public:
    NeuralNetwork(uint32_t numInputs, 
                  uint32_t numHiddenLayers, 
                  const std::vector<uint32_t>& numHiddenNeurons, 
                  uint32_t numOutputs);

    std::vector<arma::mat> forwardProp(arma::mat & input, 
                                       arma::mat (*activationFunction)(const arma::mat&));


    void train(arma::mat & inputs, 
               arma::mat & target, 
               double learningRate, 
               uint32_t epochs, 
               arma::mat (*activationFunction)(const arma::mat&), 
               arma::mat (*derivativeFunction)(const arma::mat&));

    void randomize();
    void printNetwork() const;

    ~NeuralNetwork();

private:
    std::vector<Layer> layers_;
    uint32_t numInputs_;
    uint32_t numHiddenLayers_;
    std::vector<uint32_t> numHiddenNeurons_;
    uint32_t numOutputs_;
};

#endif /* NEURALNETWORK_HPP */
