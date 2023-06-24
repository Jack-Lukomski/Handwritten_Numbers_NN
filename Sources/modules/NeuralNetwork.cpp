#include "../include/NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(NeuralNetArch_t & architecture, ActivationType af)
{
    _arch = architecture;
    _af = af;
    _layerCount = architecture.size()-1;
    _activations.push_back(arma::mat(1, architecture[0], arma::fill::zeros));

    for (size_t i = 0; i < _layerCount; ++i) {
        _weights.push_back(arma::mat(_activations[i].n_cols, architecture[i + 1], arma::fill::zeros));
        _biases.push_back(arma::mat(1, architecture[i + 1], arma::fill::zeros));
        _activations.push_back(arma::mat(1, architecture[i + 1], arma::fill::zeros));
    }
}

void NeuralNetwork::forwardProp()
{
    A_Func actFunc(_af);
    for (size_t i = 0; i < _layerCount; ++i) {
        _activations[i + 1] = _activations[i] * _weights[i];
        _activations[i + 1] += _biases[i];
        actFunc.apply(_activations[i + 1]);
    }
}

// NeuralNetwork NeuralNetwork::getGradient(std::vector<arma::mat> inputs, std::vector<arma::mat> outputs, float eps)
// {
//     NeuralNetwork gradient(_arch);
//     return gradient;
// }

float NeuralNetwork::getCost(const std::vector<arma::mat> & inputs, const std::vector<arma::mat> & outputs)
{
    assert(inputs.size() == outputs.size());

    float cost = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        NeuralNetwork::setInput(inputs[i]);
        NeuralNetwork::forwardProp();
        arma::mat difference = NeuralNetwork::getOutput() - outputs[i];
        difference = difference % difference;
        cost += arma::accu(difference);
    }
    return cost/inputs.size();
}

void NeuralNetwork::setInput(const arma::mat & input)
{
    assert(input.n_rows == _activations[0].n_rows);
    assert(input.n_cols == _activations[0].n_cols);

    _activations[0] = input;
}

arma::mat NeuralNetwork::getOutput()
{
    return _activations[_layerCount];
}

void NeuralNetwork::randomize(float min, float max)
{
    for (size_t i = 0; i < _layerCount; ++i) {
        _weights[i] = (arma::randu<arma::mat>(_weights[i].n_rows, _weights[i].n_cols) * (max - min) + min);
        _biases[i] = (arma::randu<arma::mat>(_biases[i].n_rows, _biases[i].n_cols) * (max - min) + min);
    }
}

void NeuralNetwork::print() const 
{
    for (size_t i = 0; i < _layerCount; ++i) {
        std::cout << _weights[i] << "\n" << _biases[i] << std::endl;
    }
}