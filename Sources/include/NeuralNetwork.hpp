#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "Layer.hpp"
#include <vector>
#include <functional>
#include <armadillo>

class NeuralNetwork {
public:
    NeuralNetwork(uint32_t numInputs, 
                  uint32_t numHiddenLayers, 
                  const std::vector<uint32_t>& numHiddenNeurons, 
                  uint32_t numOutputs);

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
