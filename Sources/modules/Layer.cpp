#include "../include/Layer.hpp"

Layer::Layer() {}

Layer::Layer(const arma::mat& layerWeights, const arma::mat& layerBiases, LayerType layerType)
{
    if (layerType == LayerType::InputLayer)
    {
        throw std::runtime_error("ERROR: Input Layer cannot have biases");
    }

    layerWeights_ = layerWeights;
    layerBiases_  = layerBiases;
    layerType_    = layerType;
}

Layer::Layer(const arma::mat& layerWeights, LayerType layerType)
{
    if (layerType != LayerType::InputLayer)
    {
        throw std::runtime_error("ERROR: Only the Input Layer cannot have biases");
    }

    if (layerWeights.n_cols != 1)
    {
        throw std::runtime_error("ERROR: The Input Layer can only have one col");
    }

    layerWeights_ = layerWeights;
    layerType_    = layerType;
}

const arma::mat& Layer::getLayerWeights() const
{
    return layerWeights_;
}

const arma::mat& Layer::getLayerBiases() const
{
    if (layerType_ == LayerType::InputLayer)
    {
        throw std::runtime_error("ERROR: Input Layer does not have biases");
    }

    return layerBiases_;
}

LayerType Layer::getLayerType() const
{
    return layerType_;
}

void Layer::setLayerWeights(const arma::mat& layerWeights)
{
    layerWeights_ = layerWeights;
}

void Layer::setLayerBiases(const arma::mat& layerBiases)
{
    if (layerType_ == LayerType::InputLayer)
    {
        throw std::runtime_error("ERROR: Input Layer does not have biases");
    }

    layerBiases_ = layerBiases;
}

void Layer::printLayer() const
{
    switch (layerType_)
    {
    case LayerType::InputLayer:
        std::cout << "Input Layer" << std::endl;
        std::cout << layerWeights_ << std::endl;
        break;
    case LayerType::HiddenLayer:
        std::cout << "Hidden Layer" << std::endl;
        std::cout << "Weights:" << std::endl;
        std::cout << layerWeights_ << std::endl;
        std::cout << "Biases:" << std::endl;
        std::cout << layerBiases_ << std::endl;
        break;
    case LayerType::OutputLayer:
        std::cout << "Output Layer" << std::endl;
        std::cout << "Weights:" << std::endl;
        std::cout << layerWeights_ << std::endl;
        std::cout << "Biases:" << std::endl;
        std::cout << layerBiases_ << std::endl;
        break;
    default:
        break;
    }
}

Layer::~Layer() {}