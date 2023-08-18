#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <arpa/inet.h>
#include "../include/NeuralNetwork.hpp"
#include "../include/A_Func.hpp"

// TODO: update afunc to support activation function derivaties

#define XOR_EXAMPLE 1
#define DUBBLE_NUM_EXAMPLE 0
#define MNNIST_EXAMPLE 0

#if XOR_EXAMPLE

NeuralNetArch_t arch = {2, 2, 2, 1};

arma::mat a = {{0, 0}};
arma::mat b = {{1, 0}};
arma::mat c = {{0, 1}};
arma::mat d = {{1, 1}};

arma::mat ao = arma::mat(1, 1, arma::fill::zeros);
arma::mat bo = arma::mat(1, 1, arma::fill::ones);
arma::mat co = arma::mat(1, 1, arma::fill::ones);
arma::mat doo = arma::mat(1, 1, arma::fill::zeros);

std::vector<arma::mat> inputs = {a, b, c, d};
std::vector<arma::mat> outputs = {ao, bo, co, doo};

int main ()
{
    NeuralNetwork nn(arch, ActivationType::SIGMOID);
    nn.randomize(0, 1);
    nn.print();
    nn.backprop(inputs, outputs, 0.1, 15000);
    std::cout << "\n\n\n\n";


    nn.print();

    for (size_t i = 0; i < inputs.size(); ++i) {
        std::cout << "Input is: " << inputs[i] << "Output should be: " << outputs[i];
        nn.setInput(inputs[i]);
        nn.forwardProp();
        std::cout << "The output is: " << nn.getOutput() << "\n\n" << std::endl;
    }

    return 0;
}

#endif

#if DUBBLE_NUM_EXAMPLE

NeuralNetArch_t arch = {1, 2, 2, 1};

std::vector<arma::mat> getInputs()
{
    std::vector<arma::mat> inputs;

    int n = 10;

    for (int i = 0; i < n; ++i) {
        arma::mat m = arma::zeros<arma::mat>(1, 1);
        double normalized_value = 1.0 / (1.0 + exp(-static_cast<double>(i)));
        m(0, 0) = normalized_value;
        inputs.push_back(m);
    }

    return inputs;
}

std::vector<arma::mat> getOutputs()
{
    std::vector<arma::mat> outputs;

    int n = 10;

    for (int i = 0; i < n; ++i) {
        arma::mat m = arma::zeros<arma::mat>(1, 1);
        double normalized_value = 1.0 / (1.0 + exp(-static_cast<double>(i*i)));
        m(0, 0) = normalized_value;
        outputs.push_back(m);
    }

    return outputs;
}

int main() 
{

    std::vector<arma::mat> inputs = getInputs();
    std::vector<arma::mat> outputs = getOutputs();

    NeuralNetwork nn(arch, ActivationType::SIGMOID);
    nn.randomize(0, 1);

    std::cout << "before cost=" << nn.getCost(inputs, outputs) << "\n";

    for (size_t i = 0; i < 10; ++i) {
        std::cout << "I: " << inputs[i] << "O: " << outputs[i];
        nn.setInput(inputs[i]);
        nn.forwardProp();
        std::cout << "R: " << nn.getOutput() << "\n\n" << std::endl;
    }

    nn.backprop(inputs, outputs, 0.1, 15000);

    std::cout << "after cost=" << nn.getCost(inputs, outputs) << "\n";

    for (size_t i = 0; i < 10; ++i) {
        std::cout << "I: " << inputs[i] << "O: " << outputs[i];
        nn.setInput(inputs[i]);
        nn.forwardProp();
        std::cout << "R: " << nn.getOutput() << "\n\n" << std::endl;
    }

    return 0;
}

#endif

#if MNNIST_EXAMPLE

// 28 x 28 = 784 input neurons
// using 16 neurons in hidden layers following 3b1b yt video
NeuralNetArch_t arch = {784, 16, 16, 1};

std::vector<arma::mat> getTrainingLabels()
{
    std::ifstream label_file("../training_data/MNIST/t10k-labels.idx1-ubyte", std::ios::binary);

    if (!label_file) {
        std::cerr << "Unable to open label file!" << std::endl;
        return std::vector<arma::mat>();
    }

    // Read the header data (magic number and number of items)
    int32_t magic_number = 0, number_of_labels = 0;
    label_file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number);

    label_file.read((char*)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = __builtin_bswap32(number_of_labels);

    std::vector<arma::mat> label_matrices;

    for (int i = 0; i < number_of_labels; i++) {
        unsigned char label;
        label_file.read((char*)&label, sizeof(label));

        arma::mat label_matrix = arma::zeros<arma::mat>(1, 10);
        label_matrix(0, label) = 1;
        label_matrices.push_back(label_matrix);
    }

    return label_matrices;
}

std::vector<arma::mat> getTrainingData()
{
    std::ifstream file("../training_data/MNIST/t10k-images.idx3-ubyte", std::ios::binary);

    if (file.is_open()) {
        int32_t magic_number = 0;
        int32_t number_of_images = 0;
        int32_t rows = 0;
        int32_t cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = __builtin_bswap32(magic_number);

        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = __builtin_bswap32(number_of_images);

        file.read((char*)&rows, sizeof(rows));
        rows = __builtin_bswap32(rows);

        file.read((char*)&cols, sizeof(cols));
        cols = __builtin_bswap32(cols);

        std::cout << "Magic Number: " << magic_number << "\n";
        std::cout << "Number of Images: " << number_of_images << "\n";
        std::cout << "Rows: " << rows << "\n";
        std::cout << "Cols: " << cols << "\n";

        std::vector<arma::mat> image_vectors;

        for(int i = 0; i < number_of_images; i++) {
            arma::mat reshaped_image(1, rows*cols); // this will store the current image

            for(int r = 0; r < rows; r++) {
                for(int c = 0; c < cols; c++) {
                    unsigned char pixel_byte = 0;
                    file.read((char*)&pixel_byte, sizeof(pixel_byte));
                    int linear_index = r * cols + c;
                    float pixel_float = static_cast<float>(pixel_byte) / 255.0f;
                    reshaped_image(0, linear_index) = pixel_float;
                }
            }
            image_vectors.push_back(reshaped_image);
        }

        file.close();

        return image_vectors;

    } else {
        std::cerr << "Unable to open the file!\n";
        return std::vector<arma::mat>();
    }
}


int main() 
{
    std::vector<arma::mat> inputs = getTrainingData();
    std::vector<arma::mat> outputs = getTrainingLabels();

    NeuralNetwork nn(arch, ActivationType::SIGMOID);
    nn.randomize(0, 1);

    std::cout << "before cost=" << nn.getCost(inputs, outputs) << "\n";

    for (size_t i = 0; i < 10; ++i) {
        std::cout << "I: " << outputs[i] << "O: " << outputs[i];
        nn.setInput(inputs[i]);
        nn.forwardProp();
        std::cout << "R: " << nn.getOutput() << "\n\n" << std::endl;
    }

    nn.backprop(inputs, outputs, 0.1, 10);

    std::cout << "after cost=" << nn.getCost(inputs, outputs) << "\n";

    for (size_t i = 0; i < 10; ++i) {
        std::cout << "I: " << outputs[i] << "O: " << outputs[i];
        nn.setInput(inputs[i]);
        nn.forwardProp();
        std::cout << "R: " << nn.getOutput() << "\n\n" << std::endl;
    }

}

#endif