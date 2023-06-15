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
    uint32_t prevLayerNeuronCt = numInputs;
    for (uint32_t layerIndex = 0; layerIndex < numHiddenLayers; layerIndex++)
    {
        arma::mat hiddenLayerWeights_mat = arma::zeros<arma::mat>(prevLayerNeuronCt, numHiddenNeurons[layerIndex]);
        arma::mat hiddenLayerBiases_mat = arma::zeros<arma::mat>(1, numHiddenNeurons[layerIndex]);
        Layer hiddenLayer(hiddenLayerWeights_mat, hiddenLayerBiases_mat, LayerType::HiddenLayer);
        layers_.push_back(hiddenLayer);
        prevLayerNeuronCt = numHiddenNeurons[layerIndex];
    }

    arma::mat outputLayerWeights_mat = arma::zeros<arma::mat>(prevLayerNeuronCt, numOutputs);
    arma::mat outputLayerBiases_mat = arma::zeros<arma::mat>(1, numOutputs);
    Layer outputLayer(outputLayerWeights_mat, outputLayerBiases_mat, LayerType::OutputLayer);
    layers_.push_back(outputLayer);
}

std::vector<arma::mat> NeuralNetwork::forwardProp(arma::mat & input, arma::mat (*activationFunction)(const arma::mat&))
{
    std::vector<arma::mat> outputs;
    arma::mat layerOutput = input;

    for (uint32_t layerIndex = 0; layerIndex < layers_.size(); layerIndex++)
    {
        layerOutput = layerOutput * layers_[layerIndex].getLayerWeights();
        layerOutput = layerOutput + layers_[layerIndex].getLayerBiases();
        layerOutput = activationFunction(layerOutput);
        outputs.push_back(layerOutput);
    }
    
    return outputs;
}

void NeuralNetwork::train(std::vector<arma::mat> & inputs, 
            std::vector<arma::mat> & targets, 
            double learningRate, 
            uint32_t epochs, 
            arma::mat (*activationFunction)(const arma::mat&), 
            arma::mat (*derivativeFunction)(const arma::mat&))
{
    for (uint32_t epoch = 0; epoch < epochs; epoch++)
    {
        for (uint32_t trainData = 0; trainData < inputs.size(); trainData++)
        {
            std::vector<arma::mat> outputs = NeuralNetwork::forwardProp(inputs[trainData], ActivationFunctions::sigmoid);
            arma::mat outputError = targets[trainData] - outputs.back();

            for (uint32_t layerIndex = layers_.size()-1; layerIndex > 0; layerIndex--)
            {
                arma::mat gradients = outputError % derivativeFunction(outputs[layerIndex]) * learningRate;
                arma::mat deltas = gradients * outputs[layerIndex-1];
                
                arma::mat oldWeights = layers_[layerIndex].getLayerWeights();

                arma::mat newLayerWeights = oldWeights + deltas.t();
                arma::mat newLayerBiases = layers_[layerIndex].getLayerBiases() + gradients;

                layers_[layerIndex].setLayerWeights(newLayerWeights);
                layers_[layerIndex].setLayerBiases(newLayerBiases);

                outputError = outputError * oldWeights.t();
            }
        }
    }
}

void NeuralNetwork::randomize()
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for (auto & layer: layers_)
    {
        if (layer.getLayerType() != LayerType::InputLayer)
        {
            arma::mat randWeights_mat(layer.getLayerWeights().n_rows, layer.getLayerWeights().n_cols);
            arma::mat randBiases_mat(layer.getLayerBiases().n_rows, layer.getLayerBiases().n_cols);
            
            randWeights_mat.imbue([&]() { return distribution(generator); });
            randBiases_mat.imbue([&]() { return distribution(generator); });

            layer.setLayerWeights(randWeights_mat);
            layer.setLayerBiases(randBiases_mat);
        }
    }
}

void NeuralNetwork::printNetwork() const 
{
    for (const auto & layer: layers_)
    {
        layer.printLayer();
    }
}

NeuralNetwork::~NeuralNetwork() {}