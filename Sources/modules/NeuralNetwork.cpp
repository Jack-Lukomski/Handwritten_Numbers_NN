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

void NeuralNetwork::backprop(const std::vector<arma::mat> & inputs, const std::vector<arma::mat> & outputs, float learnRate, unsigned int numEpochs)
{
    assert(inputs.size() == outputs.size());

    size_t n = inputs.size();

    for (unsigned int i = 0; i < numEpochs; i++) {
        for (size_t train_i = 0; train_i < n; ++train_i) {
            NeuralNetwork::setInput(inputs[train_i]);
            NeuralNetwork::forwardProp();

            arma::mat output = NeuralNetwork::getOutput();

            arma::mat prevLayerError = (output - outputs[train_i]) % (output % (1 - output)); // initially this is the outputError

            for (int layer_i = _layerCount-1; layer_i >= 0; --layer_i) {  // loop starts from last hidden layer towards input layer
                arma::mat deltaError;

                if (layer_i == _layerCount-1) {
                    deltaError = prevLayerError;  // for the last hidden layer, the deltaError is just the prevLayerError
                } else {
                    deltaError = (prevLayerError * _weights[layer_i+1].t()) % (_activations[layer_i+1] % (1 - _activations[layer_i+1]));
                }

                _weights[layer_i] -= learnRate * (_activations[layer_i].t() * deltaError);
                _biases[layer_i] -= learnRate * arma::sum(deltaError, 0); // Sum the deltas along the rows to match the biases dimensions
                prevLayerError = deltaError;
            }
        }
    }
}

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