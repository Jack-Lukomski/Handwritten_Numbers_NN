#ifndef LAYER_HPP
#define LAYER_HPP

#include <armadillo>
#include <iostream>

enum class LayerType {
    InputLayer,
    HiddenLayer,
    OutputLayer,
};

class Layer {
public:
    Layer();
    Layer(const arma::mat& layerWeights, const arma::mat& layerBiases, LayerType layerType);
    Layer(const arma::mat& layerWeights, LayerType layerType);

    const arma::mat& getLayerWeights() const;
    const arma::mat& getLayerBiases() const;
    LayerType getLayerType() const;

    void setLayerWeights(const arma::mat& layerWeights);
    void setLayerBiases(const arma::mat& layerBiases);
    void printLayer() const;

    ~Layer();

private:
    arma::mat layerWeights_;
    arma::mat layerBiases_;
    LayerType layerType_;
};

#endif /* LAYER_HPP */