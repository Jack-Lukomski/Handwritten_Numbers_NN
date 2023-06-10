#include "../include/NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(uint32_t numInputs, 
                             uint32_t numHiddenLayers, 
                             const std::vector<uint32_t>& numHiddenNeurons, 
                             uint32_t numOutputs) 
                            : numInputs_(numInputs), 
                            numHiddenLayers_(numHiddenLayers), 
                            numHiddenNeurons_(numHiddenNeurons), 
                            numOutputs_(numOutputs) 
{
    arma::mat inputLayer_mat = arma::zeros<arma::mat>(numInputs, 1);
    Layer inputLayer(inputLayer_mat, LayerType::InputLayer);
    layers_.push_back(inputLayer);

    uint32_t prevLayerNeuronCt = numInputs;
    for (uint32_t layerIndex = 0; layerIndex < numHiddenLayers; layerIndex++)
    {
        arma::mat hiddenLayerWeights_mat = arma::zeros<arma::mat>(numHiddenNeurons[layerIndex], prevLayerNeuronCt);
        arma::mat hiddenLayerBiases_mat = arma::zeros<arma::mat>(numHiddenNeurons[layerIndex], 1);
        Layer hiddenLayer(hiddenLayerWeights_mat, hiddenLayerBiases_mat, LayerType::HiddenLayer);
        layers_.push_back(hiddenLayer);
        prevLayerNeuronCt = numHiddenNeurons[layerIndex];
    }

    arma::mat outputLayerWeights_mat = arma::zeros<arma::mat>(numOutputs, prevLayerNeuronCt);
    arma::mat outputLayerBiases_mat = arma::zeros<arma::mat>(numOutputs, 1);
    Layer outputLayer(outputLayerWeights_mat, outputLayerBiases_mat, LayerType::OutputLayer);
    layers_.push_back(outputLayer);
}

void NeuralNetwork::randomize() 
{
    // Randomize implementation
}

void NeuralNetwork::printNetwork() const 
{
    for (const auto & layer: layers_)
    {
        layer.printLayer();
    }
}

NeuralNetwork::~NeuralNetwork() {}
